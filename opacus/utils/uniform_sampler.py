#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

import torch
from torch.utils.data import Sampler


class UniformWithReplacementSampler(Sampler):
    r"""
    This sampler samples elements according to the Sampled Gaussian Mechanism.
    Each sample is selected with a probability equal to ``sample_rate``.
    """

    def __init__(self, num_samples: int, sample_rate: float, generator=None):
        r"""
        Args:
            num_samples (int): number of samples to draw.
            sample_rate (float): probability used in sampling.
            generator (Generator): Generator used in sampling.
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.generator = generator
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    def __len__(self):
        return int(1 / self.sample_rate)

    def __iter__(self):
        num_batches = int(1 / self.sample_rate)
        while num_batches > 0:
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            if len(indices) != 0:
                # We only output non-empty list of indices, otherwise the dataloader is unhappy
                # This is compensated by the privacy engine
                yield indices
            num_batches -= 1




class DistributedPoissonBatchSampler(Sampler):
    """
    Distributed batch sampler.

    Each batch is sampled as follows:
        1. Shuffle the dataset (enabled by default)
        2. Split the dataset among the replicas into chunks of equal size
           (plus or minus one sample)
        3. Each replica selects each sample of its chunk independently
           with probability `sample_rate`
        4. Each replica ouputs the selected samples, which form a local batch

    The sum of the lengths of the local batches follows a Poisson distribution.
    In particular, the expected length of each local batch is:
        `sample_rate * total_size / num_replicas`
    """

    def __init__(
        self,
        total_size: int,
        sample_rate: float,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        generator=None,
    ):
        self.total_size = total_size
        self.sample_rate = sample_rate
        self.generator = generator
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )

        if self.total_size <= 0:
            raise ValueError(
                "total_size should be a positive integer "
                "value, but got total_size={}".format(self.total_size)
            )

        # Size of the local dataset specific to the current replica
        self.num_samples = self.total_size // self.num_replicas
        if self.rank < self.total_size % self.num_replicas:
            # The first replicas get an extra datapoint if necessary (balanced)
            self.num_samples += 1

        # Number of batches: same as non-distributed Poisson sampling, but each batch is smaller
        self.num_batches = int(1 / self.sample_rate)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g)  # type: ignore
        else:
            indices = torch.arange(self.total_size)  # type: ignore

        # Subset of the dataset assigned to this replica
        # NOTE: the first replicas might have 1 more sample.
        # (Different from the regular distributed loader that pads with more samples)
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Now, select a batch with Poisson subsampling
        for _ in range(self.num_batches):
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
            )
            selected_examples = mask.nonzero(as_tuple=False).reshape(-1)
            if len(selected_examples) > 0:
                yield indices[selected_examples]

    def __len__(self) -> int:
        """
        Expected number of batches.
        """
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch