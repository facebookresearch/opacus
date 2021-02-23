#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Union

import torch.nn as nn

from .base import ModuleChecker
from .errors import UnsupportableModuleError


BATCHNORM = Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm]


class BatchNormChecker(ModuleChecker):
    def __init__(self):
        super().__init__(
            [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm]
        )

    def validate_watched(self, module: BATCHNORM) -> None:
        raise UnsupportableModuleError(
            "BatchNorm cannot support training with differential privacy. "
            "The reason for it is that BatchNorm makes each sample's normalized value "
            "depend on its peers in a batch, ie the same sample x will get normalized to "
            "a different value depending on who else is on its batch. "
            "Privacy-wise, this means that we would have to put a privacy mechanism there too. "
            "While it can in principle be done, there are now multiple normalization layers that "
            "do not have this issue: LayerNorm, InstanceNorm and their generalization GroupNorm "
            "are all privacy-safe since they don't have this property."
            "We offer utilities to automatically replace BatchNorms to GroupNorms and we will "
            "release pretrained models to help transition, such as GN-ResNet ie a ResNet using "
            "GroupNorm, pretrained on ImageNet"
        )

    def fix_watched(self, module: BATCHNORM) -> nn.GroupNorm:
        return nn.GroupNorm(
            min(32, module.num_features), module.num_features, affine=True
        )
