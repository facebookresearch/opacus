#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Iterable

import torch.nn as nn
from opacus.supported_layers_grad_samplers import SUPPORTED_LAYERS

from .errors import UnsupportedModuleError
from .module_checkers import DPModelChecker


class ModelValidator:
    r"""
    Opacus only works with supported nn.Modules (refer to the developer README for more details).

    This class is responsible for validating that a model passed to Opacus can indeed be supported.
    Doing this before starting to train allows us to fail fast, avoiding setup time.
    This class can optionally operate in non-strict mode, in which case it will try to replace
    modules with compatible modules instead of raising an exception.

    Args:
        TODO

    Returns:
        TODO

    Raises:
        TODO

    """

    def __init__(self):
        self.dp_checker = DPModelChecker()
        self.extra_predicates = []

    def validate(self, module: nn.Module) -> List[BaseException]:
        errors = []
        for submodule_name, submodule in self.parametrized_named_modules(module):
            try:
                self.dp_checker.validate(module)
            except UnsupportedModuleError as e:
                e.msg = f"Submodule {submodule_name} is incompatible. " + e.msg
                errors.append(e)
            if type(submodule) not in SUPPORTED_LAYERS:
                errors.append(
                    UnsupportedModuleError(
                        f"Submodule {submodule_name} has type {type(submodule)} which is not supported in Opacus!"
                    )
                )

        if errors:
            error_msg = "\n\t".join(f"{i}. {s}" for i, s in enumerate(errors, 1))
            raise UnsupportedModuleError(
                f"A total of {len(errors)} modules do not match. Here are all errors: "
                f"\n\t{error_msg}\nYou can run the `fix_` method of this validator to replace these "
                "modules with our recommended replacements."
            )

    def is_valid(self, module: nn.Module) -> bool:
        try:
            self.validate(module)
        except UnsupportedModuleError:
            return False
        return True

    def parametrized_named_modules(self, module: nn.Module) -> Iterable[nn.Module]:
        """
        Recursively iterates over all submodules, returning those that
        have parameters (as opposed to "wrapper modules" that just organize modules).
        """
        yield from (
            (name, module)
            for name, module in module.named_modules()
            if any(p is not None for p in module.parameters(recurse=False))
        )

    def trainable_named_modules(self, module: nn.Module) -> Iterable[nn.Module]:
        """
        Recursively iterates over all submodules, returning those that
        have parameters and are trainable (ie they want a grad).
        """
        yield from (
            (name, module)
            for name, module in self.parametrized_named_modules(module)
            if any(p.requires_grad for p in module.parameters())
        )

    def fix_(self, module: nn.Module, verbose: bool = True) -> None:
        """
        Runs the checkers on `module`, and recurses on all its submodules.
        This will run IN-PLACE.

        Args:
            module: The root module to fix
            verbose: If True, it will announce whenever it's replacing something (defaults to True).

        Returns:
            None. This is done in-place.
        """

        for child_name, child in module.named_children():
            new_module = self.dp_checker.fix(child)
            if new_module is not None:
                setattr(module, child_name, new_module)
                if verbose:
                    print(f"Replaced `{child_name}` from {child} to {new_module}")
            else:
                self.fix_(child)
