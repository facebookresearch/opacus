#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import unittest
import time
import random
from unittest.main import main

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.layers import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP


PRIVACY_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def setup_and_get_device(rank, world_size, nonce=0):
    """
    Initialize the torch.distributed process group.
    If you run multiple groups in parallel or if you have zombie processes, you can add a nonce to avoid errors.
    """
    device = 0
    if sys.platform == "win32":
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method = "file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            "gloo", init_method=init_method, rank=rank, world_size=world_size
        )
        device = rank
    elif os.environ.get("SLURM_NTASKS") is not None:
        # Running on a Slurm cluster
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(7440 + nonce)
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        # The device is the local rank (if you have 2 nodes with 8 GPUs each, you will have two "cuda:0" devices)
        device = local_rank
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(
            init_method="env://",
            backend="nccl",
        )

        # Single node experiment
        device = rank
    return device


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size, weight, dp, noise_multiplier=0, max_grad_norm=1e8):
    # We don't want the 2 GPUs to work on the same examples/labels in parallel
    torch.manual_seed(rank)
    batch_size = 32
    withdp = "with" + ("out " if not dp else "")
    print(f"Running basic DDP {withdp} differential privacy example on rank {rank}.")

    device = setup_and_get_device(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(device)
    print(f"Initial weight: {model.net1.weight.data}")

    # Freeze all the parameters except one, to ensure that the noise is the same
    # (the DDP hook does not browse the layers in the same order as the naive implementation)
    model.net1.bias.requires_grad = False
    model.net2.bias.requires_grad = False
    model.net2.weight.requires_grad = False

    # Synchronize the model
    params = list(model.parameters())
    with torch.no_grad():
        for p in params:
            dist.broadcast(p.data, 0)
    # print(f"Weight after synchronization: {model.net1.weight.data}")

    if dp:
        ddp_model = DPDDP(model)
        engine = PrivacyEngine(
            ddp_model,
            batch_size=batch_size,
            sample_size=10 * batch_size,
            alphas=PRIVACY_ALPHAS,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            experimental=True,
            clip_per_layer=True,
        )
        engine.random_number_generator = engine._set_seed(0)
    else:
        ddp_model = DDP(model, device_ids=[device])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1)
    if dp:
        engine.attach(optimizer)

    # if rank == 0:
    #     print(model.net1.weight)
    optimizer.zero_grad()
    labels = torch.randn(batch_size, 5).to(device)

    outputs = ddp_model(torch.randn(batch_size, 10).to(device))
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # if rank == 0:
    #     print(model.net1.weight)

    weight.copy_(model.net1.weight.data.cpu())

    cleanup()


def demo_ddp_hook(rank, world_size, weight, dp, noise_multiplier, max_grad_norm):
    torch.manual_seed(rank)
    batch_size = 32
    withdp = "with" + ("out " if not dp else "")
    print(f"Running DDP hook {withdp} differential privacy example on rank {rank}.")

    device = setup_and_get_device(rank, world_size, nonce=1)

    # create model and move it to GPU with id rank
    model = ToyModel().to(device)
    # print(f"Initial weight: {model.net1.weight.data}")

    model.net1.bias.requires_grad = False
    model.net2.bias.requires_grad = False
    model.net2.weight.requires_grad = False

    ddp_model = DDP(model, device_ids=[device])
    # print(f"Initial weight after DDP wrapping: {ddp_model.module.net1.weight.data}")

    if dp:
        engine = PrivacyEngine(
            ddp_model,
            batch_size=batch_size,
            sample_size=10 * batch_size,
            alphas=PRIVACY_ALPHAS,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            experimental=True,
            clip_per_layer=True,
        )
        engine.random_number_generator = engine._set_seed(0)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1)
    if dp:
        engine.attach(optimizer)

    optimizer.zero_grad()
    labels = torch.randn(batch_size, 5).to(device)

    outputs = ddp_model(torch.randn(batch_size, 10).to(device))
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # if rank == 0:
    #     print(model.net1.weight)

    weight.copy_(model.net1.weight.data.cpu())

    del ddp_model
    cleanup()


def debug(rank, world_size, tensor, dp, noise_multiplier=0, max_grad_norm=1e8):
    local_rank = setup_and_get_device(rank, world_size)
    print(f"Rank: {rank},World size: {world_size}, local_rank: {local_rank}")
    tensor = tensor.to(local_rank)
    print(f"dp: {dp}")
    print(tensor)

    cleanup()


def run_function(local_function, tensor, dp, noise_multiplier=0, max_grad_norm=1e8):
    if os.environ.get("SLURM_NTASKS") is not None:
        world_size = int(os.environ.get("SLURM_NTASKS"))
        rank = int(os.environ.get("SLURM_PROCID"))
        print(f"Running on a Slurm cluster with {world_size} tasks.")

        local_function(rank, world_size, tensor, dp, noise_multiplier, max_grad_norm)
    else:
        world_size = torch.cuda.device_count()
        print(f"Spawning multiple processes on a local machine with {world_size} GPUs")

        # The rank will be passed as the first argument
        mp.spawn(
            local_function,
            args=(
                world_size,
                tensor,
                dp,
                noise_multiplier,
                max_grad_norm,
            ),
            nprocs=world_size,
            join=True,
        )
    return world_size


class GradientComputationTest(unittest.TestCase):
    def test_connection(self):
        tensor = torch.zeros(10, 10)
        world_size = run_function(debug, tensor, dp=True)

        self.assertTrue(
            world_size >= 2, f"Need at least 2 gpus but was provided only {world_size}."
        )

    def test_gradient_noclip_zeronoise(self):
        # Tests that gradient is the same with DP or with DDP
        weight_dp, weight_nodp = torch.zeros(10, 10), torch.zeros(10, 10)

        run_function(demo_basic, weight_dp, dp=True)
        run_function(demo_basic, weight_nodp, dp=False)

        self.assertTrue(torch.norm(weight_dp - weight_nodp) < 1e-7)

    def test_ddp_hook(self):
        # Tests that the DDP hook does the same thing as naive aggregation with per layer clipping
        weight_ddp_naive, weight_ddp_hook = torch.zeros(10, 10), torch.zeros(10, 10)

        run_function(
            demo_basic,
            weight_ddp_naive,
            dp=True,
            noise_multiplier=0.1,
            max_grad_norm=1.0,
        )

        run_function(
            demo_ddp_hook,
            weight_ddp_hook,
            dp=True,
            noise_multiplier=0.1,
            max_grad_norm=1.0,
        )

        # print(f"Naive DDP weight:")
        # print(weight_ddp_naive)
        # print("DDP hook weight:")
        # print(weight_ddp_hook)

        self.assertTrue(torch.norm(weight_ddp_naive - weight_ddp_hook) < 1e-7)
