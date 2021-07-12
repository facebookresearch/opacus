#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from functools import partial
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from opacus.layers.dp_lstm import DPLSTM, LSTMLinear
from opacus.utils.module_inspection import requires_grad
from opacus.utils.tensor_utils import calc_sample_norms


class UnsupportedModuleError(ValueError):
    pass


class GradSampleModule(nn.Module):
    r"""
    Extends nn.Module so that its parameter tensors have an extra field called .grad_sample.
    """
    GRAD_SAMPLERS = {}

    def __init__(self, m: nn.Module, *, batch_first=True, loss_reduction="mean"):
        super().__init__()
        self._module = m
        self.hooks_enabled = False
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self.add_hooks(loss_reduction=loss_reduction, batch_first=batch_first)

    def forward(self, x):
        return self._module(x)

    def zero_grad(self):
        self.del_grad_sample()
        super().zero_grad()

    def del_grad_sample(self):
        """
        Deletes ``.grad_sample`` from this module's parameters.

        Why del? Normally, ``zero_grad()`` would do ``p.grad.zero_()`` and keep the allocation.
        Normal grads can do this, because their shape is always the same.
        Grad samples do not behave like this, because they accumulate over the batch dim.
        If you have ``batch_size=32`` and size (12, 16) and you backprop twice, you should
        expect to have grad_samples of size [64, 12, 16]. If you backprop once more,
        then you'll get size [96, 12, 16] and so on.
        So when you zero out, you should be left with nothing so you can start over.
        """
        for p in self.parameters():
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                if p.grad_sample.grad_fn is not None:
                    p.grad_sample.detach_()
                else:
                    p.grad_sample.requires_grad_(False)

                del p.grad_sample

    def to_standard_module(self) -> nn.Module:
        """
        Returns the standard nn.Module wrapped by this, eliminating all traces
        of grad samples and hooks

        Returns:
            The wrapped module
        """
        self._close()
        return self._module

    def add_hooks(self, loss_reduction: str = "mean", batch_first: bool = True) -> None:
        """
        Adds hooks to model to save activations and backprop values.
        The hooks will
        1. save activations into param.activations during forward pass
        2. compute per-sample gradients in params.grad_sample during backward pass.
        Call ``remove_hooks(model)`` to disable this.

        Args:
            model: the model to which hooks are added
            loss_type: either "mean" or "sum" depending on whether backpropped
                loss was averaged or summed over batch (default: "mean")
            batch_dim: the batch dimension (default: 0)
        """
        if hasattr(self._module, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")
        else:
            self._module.autograd_grad_sample_hooks = []
            self.autograd_grad_sample_hooks = self._module.autograd_grad_sample_hooks

        for module in self.trainable_modules():
            if type(module) in self.GRAD_SAMPLERS:
                self.autograd_grad_sample_hooks.append(
                    module.register_forward_hook(self.capture_activations_hook)
                )

                self.autograd_grad_sample_hooks.append(
                    module.register_backward_hook(
                        partial(
                            self.capture_backprops_hook,
                            loss_reduction=loss_reduction,
                            batch_first=batch_first,
                        )
                    )
                )
        self.enable_hooks()

    def add_ddp_hook(
        self, engine, loss_reduction: str = "mean", batch_first: bool = True
    ) -> None:
        """
        Special function to enable DDP support on top of a GradSampleModule
        """

        self.ddp_hook_activated = True

        # We store the number of layers for the per-layer clipping
        self.n_params = 0

        # Each layer has its own callback to clip/noise
        for p in self.parameters():
            if p.requires_grad:
                self.n_params += 1
                p.register_hook(partial(self.ddp_backward_callback, engine, p))

    def remove_hooks(self) -> None:
        """
        Removes hooks added by ``add_hooks()``
        """
        self.disable_hooks()

        if self.ddp_hook_activated:
            pass
            # TODO: remove DDP hooks

        if not hasattr(self, "autograd_grad_sample_hooks"):
            raise ValueError("Asked to remove hooks, but no hooks found")
        else:
            while self.autograd_grad_sample_hooks:
                handle = self.autograd_grad_sample_hooks.pop()
                handle.remove()
            delattr(self, "autograd_grad_sample_hooks")
            delattr(self._module, "autograd_grad_sample_hooks")

    def disable_hooks(self) -> None:
        r"""
        Globally disable all hooks installed by this library.
        Why is this needed? As per https://github.com/pytorch/pytorch/issues/25723, there is
        a bug in Autograd that makes removing hooks do nothing if the graph was already
        constructed. For this reason, we have this method to at least turn them off.
        """
        self.hooks_enabled = False

    def enable_hooks(self) -> None:
        r"""
        The opposite of ``disable_hooks()``. Hooks are always enabled unless you explicitly
        disable them so you don't need to call this unless you want to re-enable them.
        """
        self.hooks_enabled = True

    def parametrized_modules(self) -> Iterable[nn.Module]:
        """
        Recursively iterates over all submodules, returning those that
        have parameters (as opposed to "wrapper modules" that just organize modules).
        """
        yield from (
            m
            for m in self._module.modules()
            if any(p is not None for p in m.parameters(recurse=False))
        )

    def trainable_modules(self) -> Iterable[nn.Module]:
        """
        Recursively iterates over all submodules, returning those that
        have parameters and are trainable (ie they want a grad).
        """
        yield from (
            m
            for m in self.parametrized_modules()
            if any(p.requires_grad for p in m.parameters())
        )

    def __repr__(self):
        return f"GradSample({self._module.__repr__()})"

    def _close(self):
        self.del_grad_sample()
        self.remove_hooks()

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        if (
            not requires_grad(module)
            or not module.training
            or not torch.is_grad_enabled()
        ):
            return

        if not self.hooks_enabled:
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append(forward_input[0].detach())  # pyre-ignore

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        """Captures backprops in backward pass and store per-sample gradients."""
        if not self.hooks_enabled:
            return

        backprops = forward_output[0].detach()
        activations, backprops = self.rearrange_grad_samples(
            module, backprops, loss_reduction, batch_first
        )
        grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
        grad_sampler_fn(module, activations, backprops)

        if (
            not isinstance(module.activations, list) or len(module.activations) == 0
        ) and hasattr(module, "max_batch_len"):
            del module.max_batch_len

    def rearrange_grad_samples(
        self,
        module: nn.Module,
        backprops: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rearrange activations and grad_samples based on loss reduction and batch dim

        Args:
            module: the module for which per-sample gradients are computed
            backprops: the captured backprops
            loss_reduction: either "mean" or "sum" depending on whether backpropped
                loss was averaged or summed over batch
            batch_first: True is batch dimension is first
        """
        if not hasattr(module, "activations"):
            raise ValueError(
                f"No activations detected for {type(module)},"
                " run forward after add_hooks(model)"
            )

        batch_dim = 0 if batch_first or type(module) is LSTMLinear else 1

        if isinstance(module.activations, list):
            A = module.activations.pop()
        else:
            A = module.activations

        if not hasattr(module, "max_batch_len"):
            # For packed sequences, max_batch_len is set in the forward of the model (e.g. the LSTM)
            # Otherwise we infer it here
            module.max_batch_len = _get_batch_size(module, A, batch_dim)

        n = module.max_batch_len
        if loss_reduction == "mean":
            B = backprops * n
        elif loss_reduction == "sum":
            B = backprops
        else:
            raise ValueError(
                f"loss_reduction = {loss_reduction}. Only 'sum' and 'mean' losses are supported"
            )

        # No matter where the batch dimension was, .grad_samples will *always* put it in the first dim
        if batch_dim != 0:
            A = A.permute([batch_dim] + [x for x in range(A.dim()) if x != batch_dim])
            B = B.permute([batch_dim] + [x for x in range(B.dim()) if x != batch_dim])

        return A, B

    @classmethod
    def is_supported(cls, module: nn.Module) -> bool:
        """Check if this module is supported"""
        return type(module) in cls.GRAD_SAMPLERS or type(module) is DPLSTM

    def ddp_backward_callback(self, engine, p, grad):
        """
        This hook operates like PrivacyEngine.step(), but on a single layer:
        1. clip_and_accumulate
        2. get the clip_values with clipper.pre_step()
        3. add the noise
        """

        # Get the norm of the gradient for each sample for this layer
        all_norms = calc_sample_norms(
            named_params=((None, p.grad_sample),),
            flat=False,
        )

        # Get the constant clipping factor for a single layer
        clipping_factor = engine.clipper.norm_clipper.calc_clipping_factors(all_norms)
        assert len(clipping_factor) == 1
        clip_factor = clipping_factor[0]
        batch_size = p.grad_sample.shape[0]

        # Do the clipping
        summed_grad = engine.clipper._weighted_sum(clip_factor, p.grad_sample)

        # accumulate the summed gradient for this mini-batch
        if hasattr(p, "summed_grad"):
            p.summed_grad += summed_grad
        else:
            p.summed_grad = summed_grad

        del p.grad_sample

        # Average (or sum) across the batch
        # TODO: add support for Poisson batch sampling
        res = engine.clipper._scale_summed_grad(p.summed_grad, batch_size)

        del p.summed_grad

        # NOTE: `self.clipper.pre_step()` returns a tensor with n=n_params clip_values
        # where each clip_value is the L2 norm of `threshs`:
        #   `max_norm = threshs.new_full((n,), threshs.norm(2))`
        # With per layer clipping, `threshs` is a tensor with n flat_values
        assert len(engine.clipper.norm_clipper.thresholds) == 1
        clipping_thresh = engine.clipper.norm_clipper.thresholds[0]
        clip_value = (self.n_params ** 0.5) * clipping_thresh
        # if engine.rank == 0:
        #     print(
        #         f"max_grad_norm: {engine.max_grad_norm} \nclipping_tresh: {clipping_thresh}\nclip_value: {clip_value}"
        #     )

        noise = _generate_noise_ddp(engine, clip_value, res)
        if engine.loss_reduction == "mean":
            noise /= batch_size

        # Only one GPU adds noise
        if engine.rank == 0:
            res += noise

        return res


def _generate_noise_ddp(
    engine, max_grad_norm: float, reference: nn.parameter.Parameter
) -> torch.Tensor:
    r"""
    Generates a tensor of Gaussian noise of the same shape as ``reference``.

    The generated tensor has zero mean and standard deviation
    sigma = ``noise_multiplier x max_grad_norm ``

    Args:
        max_grad_norm : The maximum norm of the per-sample gradients.
        reference : The reference, based on which the dimention of the
            noise tensor will be determined

    Returns:
        the generated noise with noise zero and standard
        deviation of ``noise_multiplier x max_grad_norm ``
    """
    if engine.noise_multiplier > 0 and max_grad_norm > 0:
        return torch.normal(
            0,
            engine.noise_multiplier * max_grad_norm,
            reference.shape,
            device=engine.device,
            generator=engine.random_number_generator,
        )
    return torch.zeros(reference.shape, device=engine.device)


def _get_batch_size(
    module: nn.Module, grad_sample: torch.Tensor, batch_dim: int
) -> int:
    r"""
    Computes and returns the maximum batch size which is the maximum of the dimension values
    along 'batch_dim' axis over module.activations + [grad_sample], where module.activations is
    a list. If module.activations is a not a list, then return grad_sample.shape[batch_dim].
    """

    max_batch_len = 0
    if isinstance(module.activations, list):
        for out in module.activations:
            if out.shape[batch_dim] > max_batch_len:
                max_batch_len = out.shape[batch_dim]

    max_batch_len = max(max_batch_len, grad_sample.shape[batch_dim])
    return max_batch_len
