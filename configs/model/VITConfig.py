from dataclasses import dataclass, field

@dataclass
class VITConfig:
    """
    Configuration for Vision Transformer
    """
    hf_model_name: str = "google/siglip2-base-patch16-512"
    image_size: int = 512
    patch_size: int = 16