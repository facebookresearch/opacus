#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch.nn as nn
from opacus.layers import DPMultiheadAttention

from .base import ModuleChecker
from .errors import ShouldReplaceModuleError


class MultiheadAttentionChecker(ModuleChecker):
    def __init__(self):
        super().__init__([nn.MultiheadAttention])

    def validate_watched(self, module: nn.MultiheadAttention) -> None:
        raise ShouldReplaceModuleError(
            "We do not support nn.MultiheadAttention because its implementation uses special "
            "modules. We have written a DPMultiheadAttention class that is a drop-in replacement "
            "which is compatible with our Grad Sample hooks. Please run the recommended "
            "replacement!"
        )

    def fix_watched(self, module: nn.MultiheadAttention) -> DPMultiheadAttention:
        dp_attn = DPMultiheadAttention(
            embed_dim=module.embed_dim,
            num_heads=module.num_heads,
            dropout=module.dropout,
            bias=module.in_proj_bias is not None,
            add_bias_kv=module.bias_k is not None,
            add_zero_attn=module.add_zero_attn,
            kdim=module.kdim,
            vdim=module.vdim,
        )
        dp_attn.load_state_dict(module.state_dict())
        return dp_attn
