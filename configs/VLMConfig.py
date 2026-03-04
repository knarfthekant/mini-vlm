from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class VLMConfig:
    # Language Model
    lm_model_name: str = "Qwen/Qwen3-0.6B"
    lm_tokenizer_name: str = "Qwen/Qwen3-0.6B"
    lm_config_dict: Optional[Dict[str, Any]] = None
    lm_hidden_size: int = 1024
    lm_use_tokens: bool = False
    lm_max_length: int = 4096
    
        
    # Vision Model
    vit_model_name: str = "google/siglip2-base-patch16-512"
    vit_config_dict: Optional[Dict[str, Any]] = None
    vit_hidden_size: int = 768   
    vit_image_size: int = 512 # the input image size
    vit_patch_size: int = 16 # the size of one feature patch

    # Image settings
    max_img_size: int = 1536
    resize_to_max_side_len: bool = True
    
    # Modality Projector
    pixel_shuffle_factor: int = 4
    image_token_length: int = 64
        
    # Tokenizer/Chat
    assistant_prefix: str = "<|im_start|>assistant\n"
    assistant_suffix: str = "<|im_end|>"
    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {"image_token": "<|image|>", "global_image_token": "<|global_image|>",
      "r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>", "r1c5": "<row_1_col_5>", "r1c6": "<row_1_col_6>", "r1c7": "<row_1_col_7>", "r1c8": "<row_1_col_8>",
      "r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>", "r2c5": "<row_2_col_5>", "r2c6": "<row_2_col_6>", "r2c7": "<row_2_col_7>", "r2c8": "<row_2_col_8>",
      "r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>", "r3c5": "<row_3_col_5>", "r3c6": "<row_3_col_6>", "r3c7": "<row_3_col_7>", "r3c8": "<row_3_col_8>",
      "r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>", "r4c5": "<row_4_col_5>", "r4c6": "<row_4_col_6>", "r4c7": "<row_4_col_7>", "r4c8": "<row_4_col_8>",
      "r5c1": "<row_5_col_1>", "r5c2": "<row_5_col_2>", "r5c3": "<row_5_col_3>", "r5c4": "<row_5_col_4>", "r5c5": "<row_5_col_5>", "r5c6": "<row_5_col_6>", "r5c7": "<row_5_col_7>", "r5c8": "<row_5_col_8>",
      "r6c1": "<row_6_col_1>", "r6c2": "<row_6_col_2>", "r6c3": "<row_6_col_3>", "r6c4": "<row_6_col_4>", "r6c5": "<row_6_col_5>", "r6c6": "<row_6_col_6>", "r6c7": "<row_6_col_7>", "r6c8": "<row_6_col_8>",
      "r7c1": "<row_7_col_1>", "r7c2": "<row_7_col_2>", "r7c3": "<row_7_col_3>", "r7c4": "<row_7_col_4>", "r7c5": "<row_7_col_5>", "r7c6": "<row_7_col_6>", "r7c7": "<row_7_col_7>", "r7c8": "<row_7_col_8>",
      "r8c1": "<row_8_col_1>", "r8c2": "<row_8_col_2>", "r8c3": "<row_8_col_3>", "r8c4": "<row_8_col_4>", "r8c5": "<row_8_col_5>", "r8c6": "<row_8_col_6>", "r8c7": "<row_8_col_7>", "r8c8": "<row_8_col_8>"})
    lm_chat_template: str = "{% for message in messages %}{{'1' + message['role'] + '\n' + message['content'] + '...</' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '1assistant\n' }}{% endif %}"

    # Checkpoint
    vlm_checkpoint_path: str = 'checkpoints'
    vlm_load_backbone_weights: bool = True

    # Hugging Face
    hf_repo_name: str = None

