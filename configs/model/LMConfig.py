from dataclasses import dataclass, field

@dataclass
class LMConfig:
    """
    Configuration for Language Model
    """
    hf_model_name: str = "Qwen/Qwen3-0.6B"
    tokenizer_name: str = "Qwen/Qwen3-0.6B"
    hidden_size: int = 1024
    max_position_embeddings: int = 32768

