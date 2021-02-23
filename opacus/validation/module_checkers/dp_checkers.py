#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from warnings import warn

from .batch_norm import BatchNormChecker
from .instance_norm import InstanceNormChecker
from .lstm import LSTMChecker
from .multihead_attention import MultiheadAttentionChecker


class DPModelChecker:
    """
    Runs a collection of ModuleCheckers to validate and fix a model.
    """

    def __init__(self):
        self.DP_CHECKERS = [
            BatchNormChecker(),
            InstanceNormChecker(),
            LSTMChecker(),
            MultiheadAttentionChecker(),
        ]

    def validate(self, module):
        for dp_checker in self.DP_CHECKERS:
            dp_checker.validate(module)

    def fix(self, module):
        results = [dp_checker.fix(module) for dp_checker in self.DP_CHECKERS]
        if len(results) > 1:
            warn(
                f"{len(results)} fixes returned for module {module}. Returning the first."
            )
        return results[0]
