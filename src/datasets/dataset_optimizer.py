import torch
from torch.utils.data import IterableDataset, get_worker_info
import threading
from queue import Queue
from typing import Iterator
import itertools
import random
from .training_dataset import TrainingDataset

random.seed(42)

class ConstantLengthDatasets(IterableDataset):
    """
    The pytorch dataset
    """
    def __init__(
        self,
        dataset: TrainingDataset,
        infinite: bool = False,
        max_sample_length: int = 1024,
        seq_length: int = 1024, # fixed total number of tokens that each packed sequence must contain
        num_of_sequences: int = 1024,
        queue_size: int = 2,
        max_images_per_example: int = 4,
        max_images_per_knapsack: int = 18
    ):
       self.dataset = dataset
       self.infinite = infinite
       self.max_sample_length = max_sample_length
       self.seq_length = seq_length
       self.max_length = seq_length * number_of_sequences
       self.epoch = 0
       self.queue_size = max(queue_size, 1)
       self.max_images_per_example = max_images_per_example
       self.max_images_per_knapsack = max_images_per_knapsack
       self._sentinel = object()
       self._average_length_per_sample = (
            self.dataset.image_token_length + 198 
       ) # 198 is the average number of tokens per sample

    def __len__(self):
        """
        Returns the number of packed sequences.
        """
        return int(
            len(self.dataset) * self._average_length_per_sample / self.seq_length
        )
    
    def __iter__(self):
        """
        Returns an iterator over the dataset that yields fixed-length sequences for training.

        The iterator uses a producer-consumer pattern with a background thread to efficiently
        pre-fetch and buffer samples. The producer thread continuously reads from the base
        dataset and fills a queue, while the main thread consumes from the queue.

        The dataset is automatically sharded across workers when using num_workers > 1.

        Returns:
            Iterator[dict]: An iterator that yields training samples with the following structure:
                - input_ids: Tensor of token ids of shape (seq_length,)
                - labels: Tensor of labels of shape (seq_length,)
                - attention_mask: Tensor of attention mask of shape (seq_length,)
                - images: List of processed image tensors
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
    def make_base_iterator():
        """Return a (sharded) iterator over the underlying dataset."""
        if not hasattr(self.dataset.dataset, "__len__"):
            return self.dataset.iter_for_worker()  # with iterable datasets, each worker gets different shards
        
        all_indices = range(len(self.dataset))

        # Shard the *indices* first, before any data is fetched.
        if num_workers > 1:
            worker_indices = itertools.islice(
                all_indices, worker_id, None, num_workers
            )
        else:
            worker_indices = all_indices

        # Create an iterator that only calls __getitem__ for the assigned indices.
        def sharded_item_iterator():
            for idx in worker_indices:
                yield self.dataset[idx]

        return sharded_item_iterator()
        
        for batch_of_batches in self._producer(make_base_iterator):
            for batch in batch_of_batches:
                yield batch
       
    def _producer(
        self,
        make_iterator,  # a zero-arg lambda that returns a fresh (possibly sharded) iterator
    ):
        """Runs sequentially and yields packed batches."""
        try:
            iterator = make_iterator()
            more_examples = True

            while more_examples:
                # ------------- 1) pull raw samples until we have enough -------- #
                buffer, buffer_len = [], 0
                while buffer_len < self.max_length:
                    try:
                        sample = next(iterator)
                    except StopIteration:
                        if self.infinite:
                            iterator = make_iterator()
                            self.epoch += 1
                            print(f"Epoch {self.epoch} finished, restarting iterator")
                            continue
                        else:
                            more_examples = False
                            break

                    if sample is None:  # Ratings filtered out the sample
                        continue

                    if len(sample["input_ids"]) >= self.max_sample_length:
                        continue  # skip overly long samples
                    if len(sample["images"]) > self.max_images_per_example:
                        continue  # skip samples that exceed the image constraint

                    # Truncate to seq_length - 1 to leave room for the pad token if needed, 
                    # but actually self.seq_length is the hard limit for packing.
                    if len(sample["input_ids"]) > self.seq_length - 1:
                        sample["input_ids"] = sample["input_ids"][:self.seq_length - 1]
                        sample["attention_mask"] = sample["attention_mask"][:self.seq_length - 1]
                        sample["labels"] = sample["labels"][:self.seq_length - 1]

                    sample["input_ids"] = torch.cat(
                        [
                            sample["input_ids"],
                            torch.tensor([self.dataset.tokenizer.pad_token_id]),
                        ]
                    )
                    sample["attention_mask"] = torch.cat(
                        [sample["attention_mask"], torch.tensor([0])]
                    )
                    sample["labels"] = torch.cat([sample["labels"], torch.tensor([-100])])

                    buffer.append(sample)
                    buffer_len += len(sample["input_ids"])

                if not buffer:
                    break  # nothing left and not infinite

                # ------------- 2) run greedy knapsack & pack groups ------------ #
                groups = self._balanced_greedy_knapsack(
                    buffer,
                    self.seq_length,
                    delta=5,
                    max_images_per_knapsack=self.max_images_per_knapsack,
                )

                packed_group = []
                for g in groups:
                    packed = self._pack_one_group(g, buffer, self.seq_length)
                    packed_group.append({
                        "input_ids":      packed[0],
                        "labels":         packed[1],
                        "attention_mask": packed[2],
                        "images":         packed[3],
                    })

                if packed_group:
                    yield packed_group
        except Exception as e:
            print(f"Producer encountered an error: {e}")

    def _balanced_greedy_knapsack(
        self, buffer, L, delta=0, max_images_per_knapsack=None
    ):
        # Extract lengths and image counts from buffer
        lengths = [len(x["input_ids"]) for x in buffer]
        image_counts = [len(x["images"]) for x in buffer]

        # keep the position while sorting
        items = sorted(
            enumerate(zip(lengths, image_counts)), key=lambda x: x[1][0], reverse=True
        )

        min_knapsacks = (sum(lengths) + L - 1) // L + delta
        knapsack_load = [0] * min_knapsacks
        knapsack_image_counts = [0] * min_knapsacks
        knapsack_groups = [[] for _ in range(min_knapsacks)]

        for idx, (item_len, item_image_count) in items:
            # Find a suitable knapsack that satisfies both length and image count constraints
            suitable_knapsack = None

            # First try to find a knapsack that can fit both constraints
            for ks_id in sorted(
                range(len(knapsack_load)), key=knapsack_load.__getitem__
            ):
                length_fits = knapsack_load[ks_id] + item_len <= L
                image_fits = (
                    max_images_per_knapsack is None
                    or knapsack_image_counts[ks_id] + item_image_count
                    <= max_images_per_knapsack
                )

                if length_fits and image_fits:
                    suitable_knapsack = ks_id
                    break

            # If no existing knapsack can fit, create a new one
            if suitable_knapsack is None:
                suitable_knapsack = len(knapsack_load)
                knapsack_load.append(0)
                knapsack_image_counts.append(0)
                knapsack_groups.append([])

            knapsack_groups[suitable_knapsack].append(idx)
            knapsack_load[suitable_knapsack] += item_len
            knapsack_image_counts[suitable_knapsack] += item_image_count

        # remove the completely empty bags that the +delta heuristic created
        random.shuffle(knapsack_groups)  # Knapsacks are semi-ordered after packing, thanks Luis for noticing!
        return [g for g in knapsack_groups if g]

    def _pack_one_group(self, group_indices, batch, max_len):
        ids, lbl, am, ims = [], [], [], []

        for i in group_indices:
            ids.extend(batch[i]["input_ids"])
            lbl.extend(batch[i]["labels"])
            am.extend(batch[i]["attention_mask"])
            ims.extend(batch[i]["images"])

        # safety: assert we never overflow
        if len(ids) > max_len:
            raise ValueError(f"Packed length {len(ids)} > max_len {max_len}")

        return torch.stack(ids), torch.stack(lbl), torch.stack(am), ims