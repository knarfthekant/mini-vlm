import torch
from PIL import Image
from torch.utils.data import Dataset
from src.datasets.processor import get_image_string
from datasets import DatasetDict
import logging

class TrainingDataset(Dataset):
    def __init__(
        self, 
        dataset: Dataset,
        tokenizer, 
        image_processor, 
        image_token_length, # number of visual tokens per image patch
        *,
        assistant_prefix="<|im_start|>assistant\n",
        assistant_suffix="<|im_end|>",
        relevance_min_rating=1, 
        image_correspondence_min_rating=1, 
        visual_dependency_min_rating=1, 
        formatting_min_rating=1
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_length = image_token_length
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating
        self.assistant_prefix_ids, self.assistant_suffix_ids = self._get_marker_ids(assistant_prefix, assistant_suffix)

    def __len__(self):
        return len(self.dataset)

    def _get_marker_ids(self, assistant_prefix: str, assistant_suffix: str):
        """
        Pre-calculates the token ids for the assistant prefix and suffix.
        """
        return self.tokenizer.encode(assistant_prefix, add_special_tokens=False), self.tokenizer.encode(assistant_suffix, add_special_tokens=False)

    def _get_images(self, item):
        return item['images']
    
    def _get_text_messages(self, item, splitted_image_counts):
        messages = []
        for index, text in enumerate(item['texts']):
            try: 
                if item.get('relevance_ratings') is not None and item['relevance_ratings'][index] < self.relevance_min_rating:
                    continue
                if item.get('image_correspondence_ratings') is not None and item['image_correspondence_ratings'][index] < self.image_correspondence_min_rating:
                    continue
                if item.get('visual_dependency_ratings') is not None and item['visual_dependency_ratings'][index] < self.visual_dependency_min_rating:
                    continue
                if item.get('formatting_ratings') is not None and item['formatting_ratings'][index] < self.formatting_min_rating:
                    continue
            except Exception as e:
                logging.warning(f"Failed to process item {item}, index: {index}: {e}")
            
            messages.append({"role": "user", "content": text['user']})
            messages.append({"role": "assistant", "content": text['assistant']})
        
        if len(messages) == 0:
            return messages
        
        for msg in messages:
            if self.tokenizer.image_token in msg['content']:
                logging.warning(f"Found and removed an image token in the {msg['role']} text before adding the image string.")
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")
        
        # Important step: Append image string before text messages
        if len(splitted_image_counts) > 0:
            image_string = get_image_string(self.tokenizer, splitted_image_counts, self.image_token_length)
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def _get_input_ids_and_masks(self, messages):
        """
        Generates the loss mask, the attention mask and the input ids.
        """
        # Convert tokenized messages to input token ids and attention mask
        conversation_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        input_ids = conversation_ids["input_ids"]
        attention_mask = conversation_ids["attention_mask"]

        # set mask to 0 for now
        loss_mask = [0] * len(input_ids)
        
        prefix_len = len(self.assistant_prefix_ids)
        suffix_len = len(self.assistant_suffix_ids)

        # Match the assistant prefix and suffix tokens and set the loss mask to 1 for the assistant tokens
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + prefix_len] == self.assistant_prefix_ids:
                i += prefix_len
                start = i
                while i < len(input_ids):
                    if input_ids[i:i + suffix_len] == self.assistant_suffix_ids:
                        # Include the suffix token in the loss calculation so the model learns when to stop
                        i += suffix_len
                        for j in range(start, i):
                            loss_mask[j] = 1
                        break
                    i += 1
            else:
                i += 1
        
        return {
            "input_ids": torch.tensor(input_ids),
            "loss_mask": torch.tensor(loss_mask),
            "attention_mask": torch.tensor(attention_mask),
        }
        
    def _process_images(self, images: list[Image.Image]):
        """
        Process images, return the list of images in RGB format and the metadata for their split patches.
        """
        processed_images = []
        splitted_image_counts = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError(f"Image must be compatible with PIL Image: {image}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image, splitted_image_count = self.image_processor(image)
            if not hasattr(self.tokenizer, "global_image_token") and splitted_image_count[0]*splitted_image_count[1] == len(processed_image) - 1:
                # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
                processed_image = processed_image[1:]
            processed_images.append(processed_image)
            splitted_image_counts.append(splitted_image_count)

        return processed_images, splitted_image_counts
        
