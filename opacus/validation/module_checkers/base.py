#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from abc import ABC, abstractmethod
from typing import Sequence

import torch.nn as nn
from opacus.validation.errors import UnsupportedModuleError


class ModuleChecker(ABC):

    """
    Opacus only works with supported nn.Modules (refer to the developer README for more details).

    A ModuleChecker is responsible for checking the validity of a single "basic module" (ie, one
    that is not made of submodules) and for recommending a replacement should the module not be
    compatible. ModuleCheckers are used by ``opacus.validation.ModelValidator`` to determine if a
    complex model is compatible, and for optionally making it so.

    This class is abstract, so you should inherit it to write your own ModuleChecker.

    Args:
        watched_modules: A list of module classes that your ModuleChecker will be responsible for.
            Your module will get "subscribed" to this, and will be invoked every time a submodule is
            of this class. Example: [nn.Conv1d, nn.Conv2d].

    Abstract methods:
        1. validate_watched: if you encounter a watched module, what should you do to validate it
        2. fix_watched: if you encounter a watched module, what should you return to fix it

    Your clients will instead use the ``validate`` and ``fix`` functions, which will check if
    the module is watched first.
    """

    def __init__(self, watched_modules: Sequence[type]):
        self.watched_modules = set(watched_modules)

    @abstractmethod
    def validate_watched(self, module: nn.Module) -> None:
        """
        This method gets called by ``self.validate`` and only runs on watched modules.
        Override this to have properly made exceptions.

        Args:
            module: The module (object) to validate

        Returns:
            None

        Raises:
            Depending on what the issue is, you would want to raise one of the following errors:

            - ShouldReplaceModuleError: if this module can't be supported as-is, but we do have a
                replacement (eg nn.LSTM -> DPLSTM).
            - UnsupportableModuleError: If this module can't be supported ever.
            - NotYetSupportedModuleError: if this module could be supported, but we don't yet do.
            This is also the "catch all" error.
        """

        pass

    @abstractmethod
    def fix_watched(self, module: nn.Module) -> nn.Module:
        """
        This method gets called by ``self.fix`` and only runs on watched modules.
        Override this to return a recommended replacement for this module.

        Args:
            module: The module to replace

        Returns:
            The new module
        """
        pass

    def validate(self, module: nn.Module) -> None:
        """
        Validates a module, and raises if it is not valid.
        Override this to have properly made exceptions.

        Args:
            module: The module (object) to validate

        Returns:
            None

        Raises:
            Depending on what the issue is, you would want to raise one of the following errors:

            - ShouldReplaceModuleError: if this module can't be supported as-is, but we do have a
                replacement (eg nn.LSTM -> DPLSTM).
            - UnsupportableModuleError: If this module can't be supported ever.
            - NotYetSupportedModuleError: if this module could be supported, but we don't yet do.
            This is also the "catch all" error.
        """
        if self.is_watching(module):
            self.validate_watched(module)

    def fix(self, module: nn.Module) -> nn.Module:
        """
        Returns a recommended replacement for this module, for a module that is valid instead.

        Args:
            module: The module to replace

        Returns:
            The new module. This module never runs in-place.
        """
        if self.is_watching(module):
            return self.fix_watched(module)

    def is_valid(self, module: nn.Module) -> bool:
        """
        Validates a module, without raising an exception.

        Args:
            module: The module (object) to validate

        Returns:
            True if the module is supported, False otherwise.
        """

        try:
            self.validate(module)
        except UnsupportedModuleError:
            return False
        return True

    def is_watching(self, module: nn.Module) -> bool:
        """
        Check if ``module`` should be verified by this particular checker.

        Args:
            module: The module (object) to validate

        Returns:
            True if the module should be checked by this checker, False otherwise.
        """
        return type(module) in self.watched_modules
