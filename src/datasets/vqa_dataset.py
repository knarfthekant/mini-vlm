from src.datasets.training_dataset import TrainingDataset

class VQADataset(TrainingDataset):
    def iter_for_worker(self):
        for data in self.dataset:
            yield self._process_data(data)

    def __getitem__(self, index):
        item = self.dataset[index]
        return self._process_data(item)

    def _get_labels(self, input_ids, loss_mask):
        """
        Generate labels for causal language modeling. Fill loss_mask == 0 with -100 to ignore them in loss calculation.
        """
        labels = input_ids.clone().masked_fill(loss_mask == 0, -100)
        labels = labels.roll(-1) # shift labels for causal
        labels[-1] = -100

        return labels

    def _process_data(self, item):
        # Get images as lists
        if item['images'] is None:
           images_data = []
        else:
            images_data = item['images']
            if not isinstance(images_data, list):
                images_data = [images_data]
        
        processed_images = []
        splitted_image_counts = []
        if images_data:
            processed_images, splitted_image_counts = self._process_images(images_data)
        
        messages = self._get_text_messages(item, splitted_image_counts)

        if len(messages) == 0:
            return None
        
        masks_dict = self._get_input_ids_and_masks(messages)
        input_ids = masks_dict["input_ids"]
        loss_mask = masks_dict["loss_mask"]
        attention_mask = masks_dict["attention_mask"]
        
        labels = self._get_labels(input_ids, loss_mask)

        return  {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
