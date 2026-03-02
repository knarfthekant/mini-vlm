import sys
import os
import unittest
from unittest.mock import MagicMock
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.training_dataset import TrainingDataset
from src.datasets.vqa_dataset import VQADataset
from src.datasets.processor import get_image_string
from src.datasets.tokenizer import get_tokenizer

class MockDataset(torch.utils.data.Dataset):
    def __init__(self, size=10):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            'images': [Image.new('RGB', (224, 224), color='red')],
            'texts': [
                {'user': 'What is this?', 'assistant': 'A red image.'}
            ],
            'relevance_ratings': [5],
            'image_correspondence_ratings': [5],
            'visual_dependency_ratings': [5],
            'formatting_ratings': [5]
        }

class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.mock_dataset = MockDataset()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode.side_effect = lambda x, **kwargs: [1, 2, 3] if "start" in x else [4, 5, 6]
        self.mock_tokenizer.image_token = "<image>"
        self.mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": [10, 11, 1, 2, 3, 12, 13, 4, 5, 6, 14],
            "attention_mask": [1] * 11
        }
        self.mock_tokenizer.model_max_length = 2048
        
        self.mock_image_processor = MagicMock()
        # returns (processed_image, splitted_image_count)
        self.mock_image_processor.return_value = (torch.randn(3, 3, 224, 224), (2, 2))
        
    def test_training_dataset(self):
        td = TrainingDataset(
            dataset=self.mock_dataset,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            image_token_length=64,
        )
        self.assertEqual(len(td), 10)
        
        item = self.mock_dataset[0]
        images, splitted_counts = td._process_images(item['images'])
        self.assertEqual(len(images), 1)
        self.assertEqual(splitted_counts[0], (2, 2))
        
        messages = td._get_text_messages(item, splitted_counts)
        self.assertTrue(len(messages) > 0)
        
        inputs = td._get_input_ids_and_masks(messages)
        self.assertIn("input_ids", inputs)
        self.assertIn("loss_mask", inputs)
        self.assertIn("attention_mask", inputs)
        
    def test_vqa_dataset(self):
        vd = VQADataset(
            dataset=self.mock_dataset,
            tokenizer=self.mock_tokenizer,
            image_processor=self.mock_image_processor,
            image_token_length=64,
        )
        
        item = vd[0]
        self.assertIsNotNone(item)
        self.assertIn("images", item)
        self.assertIn("input_ids", item)
        self.assertIn("labels", item)
        
        for batch in vd.iter_for_worker():
            self.assertIsNotNone(batch)
            break

    def test_get_image_string(self):
        tokenizer = MagicMock()
        tokenizer.image_token = "<img_feature>"
        tokenizer.global_image_token = "<global_img>"
        tokenizer.r1c1 = "<r1c1>"
        tokenizer.r1c2 = "<r1c2>"
        
        img_str = get_image_string(tokenizer, [(1, 2)], 2)
        # expected: <global_img><img_feature><img_feature><r1c1><img_feature><img_feature><r1c2><img_feature><img_feature>
        self.assertIn("<global_img>", img_str)
        self.assertTrue(img_str.startswith("<global_img><img_feature><img_feature><r1c1>"))

if __name__ == '__main__':
    unittest.main()
