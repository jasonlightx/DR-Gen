import torch
from torch.utils.data import DataLoader
from .datasets import TcgaImageDataset
import math
import numpy as np


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split


        if self.dataset_name == 'TCGA':
            self.dataset = TcgaImageDataset(self.args, self.tokenizer, self.split, transform=None)

        if split == 'train':
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, shuffle=self.shuffle
            )
        else:
            self.sampler = SequentialDistributedSampler(self.dataset, self.args.batch_size)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': False,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'sampler': self.sampler
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths, attr_labels, attr_masks = zip(*data)


        max_patch_len = max([img.shape[0] for img in images])
        img_dim = images[0].shape[1]

        batch_images = torch.zeros(len(images), max_patch_len, img_dim)

        for i, img in enumerate(images):
            end = img.shape[0]
            batch_images[i, :end, :] = img
        # ---------------------------------------------

        reports_ids = torch.LongTensor(np.array(reports_ids))
        reports_masks = torch.FloatTensor(np.array(reports_masks))

        attr_labels = torch.stack(attr_labels, dim=0)
        attr_masks = torch.stack(attr_masks, dim=0)

        return (images_id,
                batch_images,
                reports_ids,
                reports_masks,
                attr_labels,
                attr_masks)


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices += [indices[-1]] * (self.total_size - len(indices))
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples