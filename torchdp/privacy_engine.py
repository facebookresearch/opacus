#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import types
from typing import List, Union

import torch
from torch import nn

from . import autograd_grad_sample
from . import privacy_analysis as tf_privacy
from .dp_model_inspector import DPModelInspector
from .per_sample_gradient_clip import PerSampleGradientClipper


class PrivacyEngine:
    def __init__(
        self,
        module: nn.Module,
        batch_size: int,
        sample_size: int,
        alphas: List[float],
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        grad_norm_type: int = 2,
        batch_dim: int = 0,
        target_delta: float = 0.000001,
        **misc_settings
    ):
        self.steps = 0
        self.module = module
        self.alphas = alphas
        self.device = next(module.parameters()).device

        self.sample_rate = batch_size / sample_size
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.grad_norm_type = grad_norm_type
        self.batch_dim = batch_dim
        self.target_delta = target_delta

        self.secure_seed = int.from_bytes(os.urandom(8), byteorder="big", signed=True)
        self.secure_generator = (
            torch.random.manual_seed(self.secure_seed)
            if self.device.type == "cpu"
            else torch.cuda.manual_seed(self.secure_seed)
        )
        self.validator = DPModelInspector()
        self.clipper = None  # lazy initialization in attach
        self.misc_settings = misc_settings

    def detach(self):
        optim = self.optimizer
        optim.privacy_engine = None
        self.clipper.close()
        optim.step = types.MethodType(optim.original_step, optim)
        optim.zero_grad = types.MethodType(optim.original_zero_grad, optim)
        del optim.accumulate_grads

    def attach(self, optimizer: torch.optim.Optimizer):
        """
        Attaches to a `torch.optim.Optimizer` object, and injects itself into
        the optimizer's step.

        To do that, this method does the following:
        1. Validates the model for containing un-attachable layers
        2. Adds a pointer to this object (the PrivacyEngine) inside the optimizer
        3. Moves the original optimizer's `step()` and `zero_grad()` functions to 
           `original_step()` and `original_zero_grad()`
        4. Monkeypatches the optimizer's `step()` and `zero_grad()` functions to 
           call `step()` or `zero_grad()` on the query engine automatically 
           whenever it would call `step()` or `zero_grad()` for itself
        """

        # Validate the model for not containing un-supported modules.
        self.validator.validate(self.module)
        # only attach if model is validated
        self.clipper = PerSampleGradientClipper(
            self.module, self.max_grad_norm, self.batch_dim, **self.misc_settings
        )

        def dp_step(self, closure=None):
            self.privacy_engine.step()
            self.original_step(closure)

        def zero_all_grads(self):
            self.privacy_engine.zero_grad()
            self.original_zero_grad()

        def accumulate_grads(self):
            self.privacy_engine.accumulate_grads()

        optimizer.privacy_engine = self
        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)
        optimizer.original_zero_grad = optimizer.zero_grad
        optimizer.zero_grad = types.MethodType(zero_all_grads, optimizer)
        optimizer.accumulate_grads = types.MethodType(accumulate_grads, optimizer)

        self.optimizer = optimizer  # create a cross reference for detaching

    def get_renyi_divergence(self):
        rdp = torch.tensor(
            tf_privacy.compute_rdp(
                self.sample_rate, self.noise_multiplier, 1, self.alphas
            )
        )
        return rdp

    def get_privacy_spent(self, target_delta: float = None):
        if target_delta is None:
            target_delta = self.target_delta
        rdp = self.get_renyi_divergence() * self.steps
        return tf_privacy.get_privacy_spent(self.alphas, rdp, target_delta)

    def step(self):
        self.steps += 1
        clip_values, batch_size = self.clipper.step()
        params = (p for p in self.module.parameters() if p.requires_grad)
        for p, clip_value in zip(params, clip_values):
            noise = (
                torch.normal(
                    0,
                    self.noise_multiplier * clip_value,
                    p.grad.shape,
                    device=self.device,
                    generator=self.secure_generator,
                )
                if self.noise_multiplier > 0
                else 0.0
            )
            p.grad += noise / batch_size

    def to(self, device):
        self.device = device
        return self

    def zero_grad(self):
        autograd_grad_sample.clear_grad_sample(self.module)
        self.clipper.zero_grads()

    def accumulate_grads(self):
        self.clipper.accumulate_grads()
        # reset the accumulation of per-sample gradients
        autograd_grad_sample.clear_grad_sample(self.module)
