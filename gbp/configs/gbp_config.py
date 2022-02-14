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

from functools import partial
from typing import Optional, Tuple, Callable

from dataclasses import dataclass, field, MISSING

from gbp import logger
from gbp.utils import str2bool, get_activations, dictionary_to_option, ChoiceEnum


class Config:
    """ """

    def __init__(self, **kwargs):

        self._from_kwargs(self, kwargs)

    @classmethod
    def from_kwargs(cls, kwargs):
        self = cls()

        cls._from_kwargs(self, kwargs)

        return self

    @staticmethod
    def _from_kwargs(instance, kwargs):
        keys = set(instance.__dataclass_fields__.keys())
        # print(keys)
        # print(kwargs)
        for k, v in kwargs.items():
            # Check if the key is a valid field for configuration
            if k in keys:
                keys.remove(k)
                setattr(instance, k, v)

        if len(keys) > 0:
            logger.warning("Invalid configuration keys: {}".format(keys))

    @classmethod
    def new(cls, old, **kwargs):
        config = cls(**old.__dict__)

        for k, v in kwargs.items():
            # Check if the key is a valid field for configuration
            if hasattr(config, k):
                setattr(config, k, v)
            else:
                logger.warning("Invalid configuration key: {}".format(k))

        return config

    def duplicate(self, **kwargs):
        """
        Duplicates the configuration and overrides the provided values.
        Args:
            **kwargs: Update parameters

        Returns:
            New configuration with the updated values.
        """
        cls = type(self)
        config = cls(**self.__dict__)

        for k, v in kwargs.items():
            # Check if the key is a valid field for configuration
            if hasattr(config, k):
                setattr(config, k, v)
            else:
                logger.warning("Invalid configuration key: {}".format(k))

        return config

    def gets(self, keys):
        return {i: self.__dict__[i] for i in keys}


@dataclass(init=False)
class GBPConfig(Config):
    scalar_gate: int = 0
    vector_gate: bool = False
    vector_residual: bool = False

    scalar_act: Optional[Callable] = field(default="relu", metadata={"help": "activation function to use for scalars"})
    vector_act: Optional[Callable] = field(default="", metadata={"help": "activation function to use for vectors"})

    bottleneck: int = 1

    scalar_only: bool = True
    vector_linear: bool = True
    vector_identity: bool = True


    @property
    def activations(self) -> Tuple[Callable, Optional[Callable]]:
        return self.scalar_act, self.vector_act

    @activations.setter
    def activations(self, v: Optional[Callable]) -> None:
        self.scalar_act = v[0]
        self.vector_act = v[1]


@dataclass(init=False)
class CPDFeatures(Config):
    dihedral: bool = True
    orientations: bool = True
    sidechain: bool = True,
    relative_distance: bool = True
    relative_position: bool = True
    direction_unit: bool = True


@dataclass(init=False)
class GBPConvConfig(Config):
    edge_encoder: bool = False
    edge_gate: bool = False
    n_message: int = 3
    message_residual: int = 0
    message_ff_multiplier: int = 1
    self_message: bool = True


@dataclass(init=False)
class GBPConvLayerConfig(Config):
    gbp_conv: GBPConvConfig = field(default=MISSING, init=True)
    pre_norm: bool = False
    n_feedforward: int = 2
    drop_rate: float = 0.1


@dataclass(init=False)
class ModelConfig(Config):
    vertex_scalar: int = 100
    vertex_vector: int = 16
    edge_scalar: int = 32
    edge_vector: int = 1

    encoder_layers: int = 3
    decoder_layers: int = 3
    gnn_depth: int = 1
    dense_dropout: float = 0.1
