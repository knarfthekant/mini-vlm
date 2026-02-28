import torch
import torch.nn as nn
from transformers import AutoModel

from transformers import AutoModel, AutoConfig
from typing import Optional

class VisionTransformer(nn.Module):
    def __init__(self, config: Optional[AutoConfig] = None, hf_model_name: str = "google/siglip2-base-patch16-512"):
        super().__init__()
        if config:
            self.model = AutoModel.from_config(config)
        else:
            self.model = AutoModel.from_pretrained(hf_model_name)
        
        # Use the vision encoder for feature extraction
        if hasattr(self.model, "vision_model"):
            self.vision_model = self.model.vision_model
        else:
            self.vision_model = self.model
            
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args: pixel_values: (B, C, H, W)
        Returns: patch_embeddings: (B, L, D) where L is the number of patches
        """
        outputs = self.vision_model(pixel_values=pixel_values) 
        return outputs.last_hidden_state

    @property
    def hidden_size(self):
        return self.vision_model.config.hidden_size