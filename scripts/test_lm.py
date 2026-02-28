import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.language_model import LanguageModel
from configs.model.LMConfig import LMConfig

def test_lm():
    print("Testing LanguageModel initialization...")
    config = LMConfig()
    # Using a smaller model for quicker verification if possible, but the user asked for Qwen3-0.6B
    # We will try to load the requested model.
    model = LanguageModel(hf_model_name=config.hf_model_name)
    model.eval()
    
    print(f"Model loaded: {config.hf_model_name}")
    print(f"Hidden size: {model.hidden_size}")
    
    # Create a dummy input tensor (batch_size, sequence_length)
    dummy_input = torch.tensor([[1, 2, 3, 4, 5]]) # Just some random token IDs
    
    print("Running forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    
    # Qwen models return a CausalLMOutputWithPast or similar
    logits = output.logits
    print(f"Logits shape: {logits.shape}")
    
    # Expected shape: (batch_size, sequence_length, vocab_size)
    assert logits.shape[0] == 1
    assert logits.shape[1] == 5
    print("Success: Logits shape is correct!")

if __name__ == "__main__":
    try:
        test_lm()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
