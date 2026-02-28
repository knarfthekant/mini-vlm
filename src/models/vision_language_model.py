import json
import os
import tempfile
from dataclasses import asdice
from typing import Optional


from utils.utils import top_k_top_p_sampling
from models.vision_transformer import ViT
from models.language_model import LM
from models.modality_projector import ModalityProjector
from configs.model.VLMConfig import VLMConfig

from data.processors import get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

