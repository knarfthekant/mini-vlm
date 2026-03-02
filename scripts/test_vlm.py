import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.models.vision_language_model as VLM
from configs.model.VLMConfig import VLMConfig

def test_vlm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing VLM...")
    cfg = VLMConfig()
    # Use smaller models for testing if possible, but here we use default config
    # loading with load_backbone=False to avoid downloading weights for a simple shape check
    model = VLM.VisionLanguageModel(cfg, load_backbone=False)
    model.to(device)
    model.eval()

    batch_size = 2
    seq_len = 32
    img_size = cfg.vit_image_size
    num_images = 1
    
    # Mock data
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    # Add an image token to input_ids to trigger image processing
    image_token_id = model.tokenizer.image_token_id
    input_ids[:, 5] = image_token_id
    
    images = torch.randn(batch_size, num_images, 3, img_size, img_size).to(device)
    attention_mask = torch.ones((batch_size, seq_len)).to(device)

    print(f"Testing forward pass with batch_size={batch_size}, seq_len={seq_len}...")
    try:
        logits, loss = model.forward(input_ids, images, attention_mask=attention_mask)
        print(f"Forward pass successful. Logits shape: {logits.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

    print(f"Testing generate with max_new_tokens=5...")
    try:
        generated_ids = model.generate(
            input_ids, 
            images, 
            attention_mask=attention_mask, 
            max_new_tokens=5,
            greedy=True
        )
        print(f"Generation successful. Generated IDs shape: {generated_ids.shape}")
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    test_vlm()
