#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from opacus.validation.errors import UnsupportedModuleError


class UnsupportableModuleError(UnsupportedModuleError):
    """
    Raised whenever there is a module we can't support ever.
    BatchNorm is the largest offender.
    """

    pass


class NotYetSupportedModuleError(UnsupportedModuleError):
    """
    Raised whenever there is a module that we don't yet support.
    This is the "catch-all": the number of modules we won't ever support
    is very short, so a priori if we don't support it now it doesn't mean
    we can't extend support later (and PRs are welcome!!).
    """

    pass


class ShouldReplaceModuleError(UnsupportedModuleError):
    """
    Raised whenever there is a module that we don't support as-is but we do support via
    replacement (and we have made a replacement ourselves).
    """

    pass
