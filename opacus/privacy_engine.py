#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import types
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from . import privacy_analysis
from .per_sample_gradient_clip import PerSampleGradientClipper
from .utils import clipping
from .validation import ModelValidator

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    alphas: List[float],
    sigma_min: Optional[float] = 0.01,
    sigma_max: Optional[float] = 10.0,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        alphas: the list of orders at which to compute RDP

    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)

    """
    eps = float("inf")
    while eps > target_epsilon:
        sigma_max = 2 * sigma_max
        rdp = privacy_analysis.compute_rdp(
            sample_rate, sigma_max, epochs / sample_rate, alphas
        )
        eps = privacy_analysis.get_privacy_spent(alphas, rdp, target_delta)[0]
        if sigma_max > 2000:
            raise ValueError("The privacy budget is too low.")

    while sigma_max - sigma_min > 0.01:
        sigma = (sigma_min + sigma_max) / 2
        rdp = privacy_analysis.compute_rdp(
            sample_rate, sigma, epochs / sample_rate, alphas
        )
        eps = privacy_analysis.get_privacy_spent(alphas, rdp, target_delta)[0]

        if eps < target_epsilon:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma


DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


class PrivacyEngine:
    r"""
    The main component of Opacus is the ``PrivacyEngine``.

    To train a model with differential privacy, all you need to do
    is to define a ``PrivacyEngine`` and later attach it to your
    optimizer before running.


    Example:
        This example shows how to define a ``PrivacyEngine`` and to attach
        it to your optimizer.

        >>> import torch
        >>> model = torch.nn.Linear(16, 32)  # An example model
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        >>> privacy_engine = PrivacyEngine(model, batch_size=batch_size, sample_size=sample_size, noise_multiplier=1.3, max_grad_norm=1.0)
        >>> privacy_engine.attach(optimizer)  # That's it! Now it's business as usual.
    """

    def __init__(
        self,
        module: nn.Module,
        *,  # As per PEP 3102, this forces clients to specify kwargs explicitly, not positionally
        batch_size: Optional[int] = None,
        sample_size: Optional[int] = None,
        max_grad_norm: Union[float, List[float]],
        noise_multiplier: Optional[float] = None,
        alphas: List[float] = DEFAULT_ALPHAS,
        secure_rng: bool = False,
        batch_first: bool = True,
        target_delta: float = 1e-6,
        target_epsilon: Optional[float] = None,
        epochs: Optional[float] = None,
        loss_reduction: str = "mean",
        **misc_settings,
    ):
        r"""
        Args:
            module: The Pytorch module to which we are attaching the privacy engine
            batch_size: Training batch size. Used in the privacy accountant.
            sample_size: The size of the sample (dataset). Used in the privacy accountant.
            alphas: A list of RDP orders
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise to
                the L2-sensitivity of the function to which the noise is added
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            secure_rng: If on, it will use ``torchcsprng`` for secure random number generation.
                Comes with a significant performance cost, therefore it's recommended that you
                turn it off when just experimenting.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor will be ``[batch_size, ..., ...]``.
            target_delta: The target delta. If unset, we will set it for you.
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            **misc_settings: Other arguments to the init
        """

        self.module = module
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.sample_rate = batch_size / sample_size

        if noise_multiplier is None:
            if target_epsilon is None or target_delta is None or epochs is None:
                raise ValueError(
                    "If noise_multiplier is not specified, (target_epsilon, target_delta, epochs) should be given to the engine."
                )
            self.noise_multiplier = get_noise_multiplier(
                target_epsilon, target_delta, self.sample_rate, epochs, alphas
            )
        else:
            self.noise_multiplier = noise_multiplier

        self.max_grad_norm = max_grad_norm
        self.alphas = alphas
        self.target_delta = target_delta
        self.secure_rng = secure_rng
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self.misc_settings = misc_settings

        self.device = next(module.parameters()).device
        self.steps = 0

        if not self.batch_size or not isinstance(self.batch_size, int):
            raise ValueError(
                f"batch_size={self.batch_size} is not a valid value. Please provide a positive integer."
            )

        if not self.sample_size or not isinstance(self.sample_size, int):
            raise ValueError(
                f"sample_size={self.sample_size} is not a valid value. Please provide a positive integer."
            )

        if self.sample_rate > 1.0:
            raise ValueError(
                f"PrivacyEngine received a dataset sample size of {sample_size} "
                f"but a batch of size {batch_size}. For correct privacy accounting "
                f"the batch size must not be greater than the sample size."
            )

        if self.noise_multiplier < 0:
            raise ValueError(
                f"noise_multiplier={self.noise_multiplier} is not a valid value. Please provide a float >= 0."
            )

        if isinstance(self.max_grad_norm, float) and self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm={self.max_grad_norm} is not a valid value. Please provide a float > 0."
            )

        if not self.target_delta:
            warnings.warn(
                "target_delta unset. Setting it to an order of magnitude less than 1/sample_size."
            )
            self.target_delta = 0.1 * (1 / self.sample_size)

        if self.secure_rng:
            try:
                import torchcsprng as csprng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e

            self.seed = None
            self.random_number_generator = csprng.create_random_device_generator(
                "/dev/urandom"
            )
        else:
            warnings.warn(
                "Secure RNG turned off. This is perfectly fine for experimentation as it allows "
                "for much faster training performance, but remember to turn it on and retrain "
                "one last time before production with ``secure_rng`` turned on."
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.seed = int.from_bytes(os.urandom(8), byteorder="big", signed=True)
                self.random_number_generator = self._set_seed(self.seed)

        self.validator = ModelValidator()
        self.validator.validate(module)
        self.clipper = None  # lazy initialization in attach

    def detach(self):
        r"""
        Detaches the privacy engine from optimizer.

        To detach the ``PrivacyEngine`` from optimizer, this method returns
        the model and the optimizer to their original states (i.e. all
        added attributes/methods will be removed).
        """
        optim = self.optimizer
        optim.privacy_engine = None
        self.clipper.close()
        optim.step = types.MethodType(optim.original_step, optim)
        del optim.virtual_step

    def attach(self, optimizer: torch.optim.Optimizer):
        r"""
        Attaches the privacy engine to the optimizer.

        Attaches to the ``PrivacyEngine`` an optimizer object,and injects
        itself into the optimizer's step. To do that it,

        1. Validates that the model does not have unsupported layers.

        2. Adds a pointer to this object (the ``PrivacyEngine``) inside the optimizer.

        3. Moves optimizer's original ``step()`` function to ``original_step()``.

        4. Monkeypatches the optimizer's ``step()`` function to call ``step()`` on
        the query engine automatically whenever it would call ``step()`` for itself.

        Args:
            optimizer: The optimizer to which the privacy engine will attach
        """

        self.validator.validate(self.module)
        norm_clipper = (
            clipping.ConstantFlatClipper(self.max_grad_norm)
            if not isinstance(self.max_grad_norm, list)
            else clipping.ConstantPerLayerClipper(self.max_grad_norm)
        )

        if self.misc_settings.get("experimental", False):
            norm_clipper = clipping._Dynamic_Clipper_(
                [self.max_grad_norm],
                self.misc_settings.get("clip_per_layer", False),
                self.misc_settings.get(
                    "clipping_method", clipping.ClippingMethod.STATIC
                ),
                self.misc_settings.get("clipping_ratio", 0.0),
                self.misc_settings.get("clipping_momentum", 0.0),
            )

        self.clipper = PerSampleGradientClipper(
            self.module,
            norm_clipper,
            self.batch_first,
            self.loss_reduction,
        )

        def dp_zero_grad(self):
            self.privacy_engine.zero_grad()
            self.original_zero_grad()

        def dp_step(self, closure=None):
            self.privacy_engine.step()
            self.original_step(closure)

        optimizer.privacy_engine = self
        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)

        optimizer.original_zero_grad = optimizer.zero_grad
        optimizer.zero_grad = types.MethodType(dp_zero_grad, optimizer)

        def virtual_step(self):
            self.privacy_engine.virtual_step()

        optimizer.virtual_step = types.MethodType(virtual_step, optimizer)

        # create a cross reference for detaching
        self.optimizer = optimizer

    def get_renyi_divergence(self):
        rdp = torch.tensor(
            privacy_analysis.compute_rdp(
                self.sample_rate, self.noise_multiplier, 1, self.alphas
            )
        )
        return rdp

    def get_privacy_spent(
        self, target_delta: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Computes the (epsilon, delta) privacy budget spent so far.

        This method converts from an (alpha, epsilon)-DP guarantee for all alphas that
        the ``PrivacyEngine`` was initialized with. It returns the optimal alpha together
        with the best epsilon.

        Args:
            target_delta: The Target delta. If None, it will default to the privacy
                engine's target delta.

        Returns:
            Pair of epsilon and optimal order alpha.
        """
        if target_delta is None:
            if self.target_delta is None:
                raise ValueError(
                    "If self.target_delta is not specified, target_delta should be set as argument to get_privacy_spent."
                )
            target_delta = self.target_delta
        rdp = self.get_renyi_divergence() * self.steps
        eps, best_alpha = privacy_analysis.get_privacy_spent(
            self.alphas, rdp, target_delta
        )
        return float(eps), float(best_alpha)

    def zero_grad(self):
        """
        Resets clippers status.

        Clipper keeps internal gradient per sample in the batch in each
        ``forward`` call of the module, they need to be cleaned before the
        next round.

        If these variables are not cleaned the per sample gradients keep
        being concatenated accross batches. If accumulating gradients
        is intented behavious, e.g. simulating a large batch, prefer
        using ``virtual_step()`` function.
        """
        if self.clipper is not None:
            self.clipper.zero_grad()

    def step(self):
        """
        Takes a step for the privacy engine.

        Notes:
            You should not call this method directly. Rather, by attaching your
            ``PrivacyEngine`` to the optimizer, the ``PrivacyEngine`` would have
            the optimizer call this method for you.

        Raises:
            ValueError: If the last batch of training epoch is greater than others.
                This ensures the clipper consumed the right amount of gradients.
                In the last batch of a training epoch, we might get a batch that is
                smaller than others but we should never get a batch that is too large

        """
        self.steps += 1
        self.clipper.clip_and_accumulate()
        clip_values, batch_size = self.clipper.pre_step()

        if batch_size > self.batch_size:
            raise ValueError(
                f"PrivacyEngine expected a batch of size {self.batch_size} "
                f"but received a batch of size {batch_size}"
            )

        if batch_size < self.batch_size:
            warnings.warn(
                f"PrivacyEngine expected a batch of size {self.batch_size} "
                f"but the last step received a batch of size {batch_size}. "
                "This means that the privacy analysis will be a bit more "
                "pessimistic. You can set `drop_last = True` in your PyTorch "
                "dataloader to avoid this problem completely"
            )

        params = (p for p in self.module.parameters() if p.requires_grad)
        for p, clip_value in zip(params, clip_values):
            noise = self._generate_noise(clip_value, p)
            if self.loss_reduction == "mean":
                noise /= batch_size
            p.grad += noise

    def to(self, device: Union[str, torch.device]):
        """
        Moves the privacy engine to the target device.

        Args:
            device : The device on which Pytorch Tensors are allocated.
                See: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

        Example:
            This example shows the usage of this method, on how to move the model
            after instantiating the ``PrivacyEngine``.

            >>> model = torch.nn.Linear(16, 32)  # An example model. Default device is CPU
            >>> privacy_engine = PrivacyEngine(model, batch_size=batch_size, sample_size=sample_size, noise_multiplier=0.8, max_grad_norm=0.5)
            >>> device = "cuda:3"  # GPU
            >>> model.to(device)  # If we move the model to GPU, we should call the to() method of the privacy engine (next line)
            >>> privacy_engine.to(device)

        Returns:
            The current ``PrivacyEngine``
        """
        self.device = device
        return self

    def virtual_step(self):
        r"""
        Takes a virtual step.

        Virtual batches enable training with arbitrary large batch sizes, while
        keeping the memory consumption constant. This is beneficial, when training
        models with larger batch sizes than standard models.

        Example:
            Imagine you want to train a model with batch size of 2048, but you can only
            fit batch size of 128 in your GPU. Then, you can do the following:

            >>> for i, (X, y) in enumerate(dataloader):
            >>>     logits = model(X)
            >>>     loss = criterion(logits, y)
            >>>     loss.backward()
            >>>     if i % 16 == 15:
            >>>         optimizer.step()    # this will call privacy engine's step()
            >>>         optimizer.zero_grad()
            >>>     else:
            >>>         optimizer.virtual_step()   # this will call privacy engine's virtual_step()

            The rough idea of virtual step is as follows:

            1. Calling ``loss.backward()`` repeatedly stores the per-sample gradients
            for all mini-batches. If we call ``loss.backward()`` ``N`` times on
            mini-batches of size ``B``, then each weight's ``.grad_sample`` field will
            contain ``NxB`` gradients. Then, when calling ``step()``, the privacy engine
            clips all ``NxB`` gradients and computes the average gradient for an effective
            batch of size ``NxB``. A call to ``optimizer.zero_grad()`` erases the
            per-sample gradients.

            2. By calling ``virtual_step()`` after ``loss.backward()``,the ``B``
            per-sample gradients for this mini-batch are clipped and summed up into a
            gradient accumulator. The per-sample gradients can then be discarded. After
            ``N`` iterations (alternating calls to ``loss.backward()`` and
            ``virtual_step()``), a call to ``step()`` will compute the average gradient
            for an effective batch of size ``NxB``.

            The advantage here is that this is memory-efficient: it discards the per-sample
            gradients after every mini-batch. We can thus handle batches of arbitrary size.
        """
        self.clipper.clip_and_accumulate()

    def _generate_noise(
        self, max_grad_norm: float, reference: nn.parameter.Parameter
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
        if self.noise_multiplier > 0 and max_grad_norm > 0:
            return torch.normal(
                0,
                self.noise_multiplier * max_grad_norm,
                reference.grad.shape,
                device=self.device,
                generator=self.random_number_generator,
            )
        return torch.zeros(reference.grad.shape, device=self.device)

    def _set_seed(self, seed: int):
        r"""
        Allows to manually set the seed allowing for a deterministic run. Useful if you want to
        debug.

        WARNING: MANUALLY SETTING THE SEED BREAKS THE GUARANTEE OF SECURE RNG.
        For this reason, this method will raise a ValueError if you had ``secure_rng`` turned on.

        Args:
            seed : The **unsecure** seed
        """
        if self.secure_rng:
            raise ValueError(
                "Seed was manually set on a ``PrivacyEngine`` with ``secure_rng`` turned on."
                "This fundamentally breaks secure_rng, and cannot be allowed. "
                "If you do need reproducibility with a fixed seed, first instantiate the PrivacyEngine "
                "with ``secure_seed`` turned off."
            )
        self.seed = seed

        return (
            torch.random.manual_seed(self.seed)
            if self.device.type == "cpu"
            else torch.cuda.manual_seed(self.seed)
        )
