import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional

class LanguageModel(nn.Module):
    def __init__(self, config: Optional[AutoConfig] = None, hf_model_name: str = "Qwen/Qwen3-0.6B"):
        super().__init__()
        if config:
            self.model = AutoModelForCausalLM.from_config(config) # Auto regressive model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    @property
    def hidden_size(self):
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        return self.model.config.d_model # Some models use d_model

      