from transformers import AutoTokenizer
import torchvision.transforms as transforms

TOKENIZERS_CACHE = {}

def get_tokenizer(name, extra_special_tokens: dict[str, str] = None, chat_template: str = None):
    """
    Build custom tokenizer for VLM
    """
    if name not in TOKENIZERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        if extra_special_tokens is not None:
            # Register special tokens
            special_tokens_list = list(extra_special_tokens.values())
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})

            # Set attributes for easy access
            for key, val in extra_special_tokens.items():
                setattr(tokenizer, key, val)
                setattr(tokenizer, f"{key}_id", tokenizer.convert_tokens_to_ids(val))

        if chat_template is not None:
            tokenizer.chat_template = chat_template

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]
            
        