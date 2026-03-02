import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, Union

class LanguageModel(nn.Module):
    def __init__(
        self, 
        config: Optional[Union[AutoConfig, str]] = "Qwen/Qwen3-0.6B"
    ):
        super().__init__()
        if isinstance(config, str):
            # download from hub
            self.model = AutoModelForCausalLM.from_pretrained(config)
        else:
            # config is an AutoConfig/PretrainedConfig object
            self.model = AutoModelForCausalLM.from_config(config)
        
        self.lm_use_tokens = False # Default to returning hidden states

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        
        if self.lm_use_tokens:
            # Return (logits, past_key_values)
            return outputs.logits, outputs.past_key_values
        else:
            # Return (last_hidden_state, past_key_values)
            return outputs.hidden_states[-1], outputs.past_key_values

    @property
    def token_embedding(self):
        return self.model.get_input_embeddings()

    @property
    def head(self):
        return self.model.get_output_embeddings()

    @property
    def hidden_size(self):
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        return self.model.config.d_model # Some models use d_model

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def set_use_tokens(self, lm_use_tokens: bool):
        self.lm_use_tokens = lm_use_tokens
        

      