#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Union

import torch.nn as nn
from opacus.utils.clone import clone_module

from .base import ModuleChecker
from .errors import UnsupportableModuleError


class InstanceNormChecker(ModuleChecker):
    def __init__(self):
        super().__init__([nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d])

    def validate_watched(
        self, module: Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    ) -> None:
        if module.track_running_stats:
            raise UnsupportableModuleError(
                "We do not support tracking running stats with differential privacy. "
                "To support it, we would have to add a DP mechanism for these statistics too "
                "which would incur a privacy cost for little value in model accuracy. "
                "Just say no to running stats :)"
            )

    def fix_watched(
        self, module: Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    ) -> Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]:
        new_module = clone_module(module)
        new_module.track_running_stats = False
        return new_module
