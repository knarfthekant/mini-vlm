from dataclasses import dataclass, field

@dataclass
class VLMConfig:
    """
    Configuration for Vision Language Model
    """
    # Language Model
    lm_model_name: str = "Qwen/Qwen3-0.6B"
    lm_tokenizer_name: str = "Qwen/Qwen3-0.6B"
    lm_hidden_size: int = 1024
    lm_max_position_embeddings: int = 32768

    # Vision Model
    vit_model_name: str = "google/siglip2-base-patch16-512"
    vit_image_size: int = 512
    vit_patch_size: int = 16
    vit_hidden_size: int = 768

    # Modality Projector
    pixel_shuffle_factor: int = 4
    image_token_length: int = 64

