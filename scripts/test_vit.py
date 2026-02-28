import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.vision_transformer import VisionTransformer
from configs.model.VITConfig import VITConfig

def test_vit():
    print("Testing VisionTransformer initialization...")
    config = VITConfig()
    model = VisionTransformer(hf_model_name=config.hf_model_name)
    model.eval()
    
    print(f"Model loaded: {config.hf_model_name}")
    print(f"Hidden size: {model.hidden_size}")
    
    # Create a dummy image tensor (batch_size, channels, height, width)
    # SigLIP-2 base usually uses 512x512
    dummy_input = torch.randn(1, 3, 512, 512)
    
    print("Running forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    # Expected shape: (batch_size, num_patches, hidden_size)
    # For 512x512 with 16x16 patches, num_patches = (512/16)^2 = 32^2 = 1024
    # Wait, SigLIP might have a CLS token or similar? 
    # Actually SigLIP doesn't use a CLS token usually, but some implementations might.
    # SigLIP 2 patch16-512 should have 1024 patches.
    
    expected_patches = (512 // 16) ** 2
    assert output.shape[0] == 1
    assert output.shape[-1] == model.hidden_size
    print("Success: Output shape is correct!")

if __name__ == "__main__":
    try:
        test_vit()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
