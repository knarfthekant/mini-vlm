import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional

from src.utils.utils import top_k_top_p_filtering
from src.models.vision_transformer import ViT
from src.models.language_model import LanguageModel
from src.models.modality_projector import ModalityProjector
from configs.VLMConfig import VLMConfig

from src.datasets.tokenizer import get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from safetensors.torch import load_model, save_model

class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        cfg: VLMConfig,
        load_backbone: bool = True
    ):
        super().__init__()
        self.cfg = cfg
        
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT(cfg.vit_model_name)
            self.decoder = LanguageModel(cfg.lm_model_name)
        else:
            # Initialize from scratch using configs
            vit_config = AutoConfig.from_pretrained(cfg.vit_model_name)
            if cfg.vit_config_dict:
                vit_config.update(cfg.vit_config_dict)
            
            lm_config = AutoConfig.from_pretrained(cfg.lm_model_name)
            if cfg.lm_config_dict:
                lm_config.update(cfg.lm_config_dict)

            self.vision_encoder = ViT(vit_config)
            self.decoder = LanguageModel(lm_config)
        
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer_name, cfg.vlm_extra_tokens, cfg.lm_chat_template)
        self.decoder.set_use_tokens(cfg.lm_use_tokens)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.decoder, 'gradient_checkpointing_enable'):
            self.decoder.gradient_checkpointing_enable(**kwargs)
        if hasattr(self.vision_encoder, 'gradient_checkpointing_enable'):
            self.vision_encoder.gradient_checkpointing_enable(**kwargs)
    
    def _fill_img_tokens(self, input_ids, token_embd, image_embd):
        """
        Args:
            token_embd: (batch_size, tokenized_seq_len, dim_lm)
            image_embd: (num_images, mp_image_token_length, dim_lm)
        """
        mask = (input_ids == self.tokenizer.image_token_id) # identify image tokens
        mask_unsqueezed = mask.unsqueeze(-1) # (batch_size, tokenized_seq_len, 1)
        # view image embeddings as (total_image_tokens, dim_lm)
        image_embd_flat = image_embd.view(-1, image_embd.shape[-1]).to(token_embd.dtype)

        # create a tensor of the same shape as token_embd but with image embeddings
        image_tokens_only = torch.zeros_like(token_embd)
        # Place image embeddings in the position of image tensors
        image_tokens_only.masked_scatter_(mask_unsqueezed, image_embd_flat)

        # combine using torch.where
        return torch.where(mask_unsqueezed, image_tokens_only, token_embd)

    def _process_images(self, images, device):
        """
        Prepare batches of images
        """
        if isinstance(images, list):

            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]

            if not images:
                return None
            else:
                return torch.cat(images, dim=0).to(device)
        
        if torch.is_tensor(images):
            if images.dim() == 5:
                # (batch_size, num_images, C, H, W) -> (batch_size * num_images, C, H, W)
                images = images.view(-1, *images.shape[-3:])
            return images.to(device)
        return images

    def forward(
            self, 
            input_ids,
            images,
            attention_mask=None, # mask for padding tokens
            targets=None # ground truth labels
        ):
        """
        Args:
            input_ids: (batch_size, tokenized_seq_len)
            images: (batch_size, num_images, 3, 512, 512)
            attention_mask: (batch_size, tokenized_seq_len)
            targets: (batch_size, tokenized_seq_len)
        """
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids) # (batch_size, tokenized_seq_len, dim_lm)

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor) 
            image_embd = self.MP(image_embd)
            token_embd = self._fill_img_tokens(input_ids, token_embd, image_embd)

        hidden_states, _ = self.decoder(None, inputs_embeds=token_embd, attention_mask=attention_mask)
        logits = self.decoder.head(hidden_states)
        loss = None
        if targets is not None:
            # prevent NaN loss when all targets are masked
            if (targets != -100).sum() == 0:
                logits_for_zero = self.decoder.head(logits)
                loss = (logits_for_zero * 0).sum()
            else:
                loss = self._chunked_cross_entropy(logits, targets, chunk_size=128)
        
        return logits, loss

    def _chunked_cross_entropy(self, hidden_states, targets, chunk_size=128):
        """
        Compute cross entropy loss in chunks to save memory
        """
        B, T, D = hidden_states.shape # (batch_size, tokenized_seq_len, dim_lm)
        total_loss = torch.tensor(0.0, device=hidden_states.device)
        num_valid_tokens = (targets != -100).sum() # count ignored tokens

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_hidden = hidden_states[:, start:end, :]
            chunk_targets = targets[:, start:end]

            with torch.amp.autocast(device_type=hidden_states.device.type, enabled=False):
                chunk_logits = self.decoder.head(chunk_hidden.float()) # (B, chunk, V)
                chunk_loss = F.cross_entropy(
                    chunk_logits.view(-1, chunk_logits.size(-1)),
                    chunk_targets.view(-1),
                    ignore_index=-100,
                    reduction="sum"
                )
                total_loss = total_loss + chunk_loss

        return total_loss / num_valid_tokens

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        images,
        attention_mask=None,
        max_new_tokens=5,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        greedy=False
    ):
        # Handle low temperature
        if temperature == 0.0:
            greedy = True
        
        images_tensor = self._process_images(images, input_ids.device) # (batch_size, 3, 512, 512)
        token_embd = self.decoder.token_embedding(input_ids) # (batch_size, tokenized_seq_len, dim_lm)

        if images_tensor is not None:
            # process images
            image_embd = self.vision_encoder(images_tensor) # (batch_size, seq_image_feat, dim_vit)
            image_embd = self.MP(image_embd) # (batch_size, num_patches, dim_lm)
            token_embd = self._fill_img_tokens(input_ids, token_embd, image_embd) 

        current_token_seq_len = token_embd.size(1)
        batch_size = token_embd.size(0)

        # Prefill phase, calculate kv cache and understands context
        hidden_states, kv_cache_list = self.decoder(
            None,
            inputs_embeds=token_embd,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True
        )

        last_token_output_from_prefill = hidden_states[:, -1, :]
        
        # LM use tokens ids or raw embeddings
        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill)
        else:
            current_logits = last_token_output_from_prefill
        
        # Store newly generated token Ids
        newly_generated_ids_list = []

        # Decoding loop, token decodes in batch
        for _ in range(max_new_tokens):
            if greedy:
                # Greedy logit selection
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                # Perform sampling
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1) # performing tempe
                next_token_id = torch.multinomial(probs, num_samples=1)
    
            newly_generated_ids_list.append(next_token_id)

            # The start_pos for the new token is the current total sequence length before adding this new token
            current_token_start_pos = current_token_seq_len
            current_token_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)

            # Process new token with KV cache
            # We need to get embedding for next_token_id
            next_token_embed = self.decoder.token_embedding(next_token_id)
            hidden_states, kv_cache_list = self.decoder(
                None,
                inputs_embeds=next_token_embed,
                attention_mask=attention_mask,
                past_key_values=kv_cache_list,
                use_cache=True
            )

            last_token_output = hidden_states[:, -1, :]

            # Apply head to get logits if model is in embedding mode
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output

        if not newly_generated_ids_list: # Hande case where max_new_tokens might be 0
            return torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)
        
        generated_ids = torch.cat(newly_generated_ids_list, dim=1) 

        # Post-process to handle EOS token
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0: # not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (generated_ids == self.tokenizer.eos_token_id)

            col_indices_for_min = torch.arange(seq_len, device=device) # create column indices

            # In eos_mask, mark positions with actual col_idx, others with large number
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1)

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values

            # Clamp values to seq_len 
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            # Create column indices for comparison, shape (batch_size, seq_len)
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)

            # Tokens are replaced if their column index is greater than the index of first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)

            generated_ids[replace_mask] = self.tokenizer.eos_token_id

        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a pre-trained VisionLanguageModel 
        """

        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.pth.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path"
                )

            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path"
                )
        else:
            from hugging_face_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )
        
        # load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))
        
        model = cls(cfg, load_backbone=False)

        load_model(model, weights_path)

        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.
        Args:
            save_directory (str): The directory to save the model and configuration to.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))
        
        save_model(self, os.path.join(save_directory, "model.safetensors"))
        
    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )
        


            


        
    


        



        
