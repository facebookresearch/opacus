#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


class UnsupportedModuleError(ValueError):
    """
    Raised whenever there is a module we can't support ever.
    BatchNorm is the largest offender.
    """

    pass
