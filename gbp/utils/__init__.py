# coding=utf-8
# Copyright 2021 GBP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""  """

from __future__ import print_function, absolute_import, division

import argparse
from enum import Enum
from typing import List

import torch
from torch.nn import functional as F

from .utils import get_logger

log = get_logger(__name__)

# https://github.com/pytorch/fairseq/blob/main/fairseq/dataclass/constants.py
class StrEnum(Enum):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


def squared_relu(x):
    return F.relu(x) ** 2


# https://github.com/sunglasses-ai/classy/blob/3e74cba1fdf1b9f9f2ba1cfcfa6c2017aa59fc04/classy/optim/factories.py#L14


def get_activations(optional=False):
    activations = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "relusq": squared_relu,
        "sigmoid": torch.nn.Sigmoid,
        "silu": torch.nn.SiLU,
        "selu": torch.nn.SELU,
    }
    if optional:
        activations[""] = None

    return activations


def get_activations_none(optional=False):
    activations = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "relusq": squared_relu,
        "sigmoid": torch.nn.Sigmoid,
        "silu": torch.nn.SiLU,
        "selu": torch.nn.SELU,
    }
    if optional:
        activations[""] = None
        activations[None] = None

    return activations


def dictionary_to_option(options, selected):
    if selected not in options:
        raise argparse.ArgumentTypeError(
            f'Invalid choice "{selected}", choose one from {", ".join(list(options.keys()))} '
        )
    return options[selected]()


def str2act(input_str):
    if input_str == "":
        return None

    act = get_activations(optional=True)
    out = dictionary_to_option(act, input_str)
    return out


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
