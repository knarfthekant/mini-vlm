import torch

class TrainingCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def prepare_batch(self, batch: list[dict], max_length: int = None):
        """
        Batch is a list of dictionaries with keys "input_ids", "labels", "attention_mask", "images"
        Convert batch to a dictionary of lists of tensors
        """
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}
        
        # Remove None rows
        batch = [x for x in batch if x is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # Convert list of dictionaries to a dictionary of lists of tensors
        batch = {k: [x[k] for x in batch] for k in batch[0]}

        if max_length is not None:
            batch = self._discard_overlong_samples(batch, max_length)

        if len(batch["input_ids"]) == 0:
            return batch

        # Pad batch
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"])) # gets the max length of the input ids

        self._pad_batch(batch, max_len)
        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "labels": torch.stack(batch["labels"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"]
        }

    def _pad_batch(self, batch: dict[list[torch.Tensor]], max_length: int):
        """
        Batch is a dictionary of lists of tensors
        Each tensor in the batch is padded to (max_length,)
        """
        # Note input ids can be multiple tensors
        batch["input_ids"] = [torch.nn.functional.pad(id, (max_length - len(ids), 0), value=self.tokenizer) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def _discard_overlong_samples(self, batch: dict[list[torch.Tensor]], max_length: int):
        filtered = [
            (ids, label, atten, img)
            for ids, label, atten, img in zip(batch["input_ids"], batch["labels"], batch["attention_mask"], batch["images"])
            if len(ids) <= max_length
        ]
        if not filtered:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # Splits the filtered list of tuples into four lists
        batch_input_ids, batch_labels, batch_attentions, batch_images = zip(*filtered)
        return {
            "input_ids": list(batch_input_ids),
            "labels": list(batch_labels),
            "attention_mask": list(batch_attentions),
            "images": list(batch_images)
        }

class VQACollator(TrainingCollator):
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)

    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length)
        return batch

