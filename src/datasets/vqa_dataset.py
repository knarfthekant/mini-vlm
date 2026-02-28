from src.datasets.training_dataset import TrainingDataset

class VQADataset(TrainingDataset):
    def iter_for_worker(self):
        for data in self.dataset:
            yield self._process_data(data)

    def __getitem__(self, index):
        item = self.dataset[idx]
        return self._process_data(item)

    def _get_labels(self, input_ids, loss_mask):
        """
        Generate labels for causal language modeling.
        """
        labels = input_ids.clone().masked_fill(~loss_mask, -100)
        labels = labels.roll(-1) # shift labels for causal
        labels[-1] = -100

        return labels

    def _process_data(self, item):
        # Get images as lists
        if item['images'] is None:
           images_data = []
        else:
            images_data = item['images']
            if not isinstance(image_data, list):
                images_data = [image_data]
        
        processed_images = []
        splitted_images_counts = []
        if images_data:
            processed_images, splitted_images_counts = self._process_images(images_data)
        
        messages = self._get_messages(item, splitted_image_counts)

        if len(messages) == 0:
            return None
        
        input_ids, loss_mask, attention_mask = self._get_input_ids_and_masks(messages)
        labels = self._get_labels(input_ids, loss_mask)

        return  {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
