import torch.nn as nn
import math

class ModalityProjector(nn.Module):
    """
    A MLP-based modality projector.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_size = cfg.vit_hidden_size * (cfg.pixel_shuffle_factor ** 2)
        self.output_size = cfg.lm_hidden_size # connecting ViT to LM
        self.scale_factor = cfg.pixel_shuffle_factor

        self.proj = nn.Linear(self.input_size, self.output_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize linear layer weights with normal distribution
            nn.init.normal_(module.weight, mean=0.0, std=module.in_features**-0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def pixel_shuffle(self, x):
        """
        Space to Depth algorithm. Reduce the number of visual tokens before they are passed to
        the LLM while preserving all the original information by increasing the embedding dimension
        Optimizes the self attention mechanism
        """
        bsz, seq, embed_dim = x.size()
        seq_root = int(math.sqrt(seq))
        assert seq_root * seq_root == seq, "Sequence length must be a perfect square"
        assert seq_root % self.scale_factor == 0, "Sequence length must be divisible by scale factor"
        
        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x