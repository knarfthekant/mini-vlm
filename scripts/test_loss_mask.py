import sys
import os
from unittest.mock import MagicMock
from typing import Callable

# Mock torch before it's imported by TrainingDataset
try:
    import torch
except ImportError:
    torch_mock = MagicMock()
    # Basic mock for torch.tensor
    def mock_tensor_func(data):
        m = MagicMock()
        m.tolist = lambda: data
        m.__len__ = lambda: len(data)
        m.__getitem__ = lambda self, idx: data[idx] if isinstance(idx, int) else data[idx] # simplified
        return m
    torch_mock.tensor = mock_tensor_func
    sys.modules["torch"] = torch_mock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.training_dataset import TrainingDataset

class MockTokenizer:
    def __init__(self):
        self.image_token = "<image>"
        self.model_max_length = 2048
        # Define some arbitrary IDs for our markers
        self.prefix = "<|im_start|>assistant\n"
        self.suffix = "<|im_end|>"
        self.prefix_ids = [100, 101, 102]
        self.suffix_ids = [200, 201]

    def encode(self, text, add_special_tokens=False):
        if text == self.prefix:
            return self.prefix_ids
        if text == self.suffix:
            return self.suffix_ids
        return [1] * len(text) # Dummy encoding

    def apply_chat_template(self, messages, **kwargs):
        # Construct a dummy input_ids sequence
        # We'll just concatenate segments
        input_ids = []
        for msg in messages:
            if msg['role'] == 'user':
                input_ids.extend([10] * 5) # User message contents
            elif msg['role'] == 'assistant':
                input_ids.extend(self.prefix_ids)
                input_ids.extend([20] * 3) # Assistant message contents
                input_ids.extend(self.suffix_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids)
        }

def test_loss_mask():
    tokenizer = MockTokenizer()
    # We pass a dummy dataset and image_processor
    dataset = TrainingDataset(
        dataset=[], 
        tokenizer=tokenizer,
        image_processor=lambda x: (x, (1, 1)),
        image_token_length=10
    )

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    outputs = dataset._get_input_ids_and_masks(messages)
    
    input_ids = outputs["input_ids"]
    loss_mask = outputs["loss_mask"]
    
    # input_ids is a MagicMock return, but we can get the data via tolist()
    input_ids_list = input_ids.tolist()
    loss_mask_list = loss_mask.tolist()
    
    print(f"Input IDs: {input_ids_list}")
    print(f"Loss Mask: {loss_mask_list}")
    
    # Prefix is [100, 101, 102], Suffix is [200, 201]
    # Expected input_ids: [10, 10, 10, 10, 10, 100, 101, 102, 20, 20, 20, 200, 201]
    # start = index of first '20' (after prefix) = 8
    # end = index after '201' (suffix) = 13
    # loss_mask should be 1 from index 8 to 12
    
    expected_mask = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    
    assert len(loss_mask_list) == len(input_ids_list), "Mask length mismatch"
    assert loss_mask_list == expected_mask, f"Expected {expected_mask}, got {loss_mask_list}"
    print("Success: Loss mask correctly identifies assistant tokens!")

    # Test multiple turns
    messages_multi = [
        {"role": "user", "content": "Talk 1"},
        {"role": "assistant", "content": "Resp 1"},
        {"role": "user", "content": "Talk 2"},
        {"role": "assistant", "content": "Resp 2"},
    ]
    
    outputs_multi = dataset._get_input_ids_and_masks(messages_multi)
    input_ids_multi = outputs_multi["input_ids"]
    loss_mask_multi = outputs_multi["loss_mask"]
    
    input_ids_multi_list = input_ids_multi.tolist()
    loss_mask_multi_list = loss_mask_multi.tolist()
    
    print(f"Multi Input IDs: {input_ids_multi_list}")
    print(f"Multi Loss Mask: {loss_mask_multi_list}")
    
    # Expected sequence:
    # [10]*5, [100,101,102], [20]*3, [200,201], [10]*5, [100,101,102], [20]*3, [200,201]
    # Indices:
    # 0-4: User 1
    # 5-7: Prefix 1
    # 8-10: Content 1
    # 11-12: Suffix 1
    # 13-17: User 2
    # 18-20: Prefix 2
    # 21-23: Content 2
    # 24-25: Suffix 2
    
    expected_mask_multi = [0]*8 + [1]*5 + [0]*8 + [1]*5
    assert loss_mask_multi_list == expected_mask_multi, f"Multi turn failed. Expected {expected_mask_multi}, got {loss_mask_multi_list}"
    print("Success: Multi-turn loss mask correctly identifies assistant tokens!")

if __name__ == "__main__":
    try:
        test_loss_mask()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
