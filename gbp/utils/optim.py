from __future__ import print_function, absolute_import, division

from abc import ABC
from typing import Optional, List

import hydra
import torch
from omegaconf import DictConfig

# https://github.com/sunglasses-ai/classy/blob/main/classy/optim/factories.py
from torch.optim import Adam

from gbp.utils.optimization import get_scheduler, get_scheduler_proxy, SchedulerConfig

from .utils import get_logger

log = get_logger(__name__)


class Factory:
    """Factory interface that allows for simple instantiation of optimizers and schedulers for PyTorch Lightning.
    This class is essentially a work-around for lazy instantiation:
    * all params but for the module to be optimized are received in __init__
    * the actual instantiation of optimizers and schedulers takes place in the __call__ method, where the module to be
      optimized is provided
    __call__ will be invoked in the configure_optimizers hooks of LighiningModule-s and its return object directly returned.
    As such, the return type of __call__ can be any of those allowed by configure_optimizers, namely:
    * Single optimizer
    * List or Tuple - List of optimizers
    * Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict)
    * Dictionary, with an ‘optimizer’ key, and (optionally) a ‘lr_scheduler’ key whose value is a single LR scheduler or lr_dict
    * Tuple of dictionaries as described, with an optional ‘frequency’ key
    * None - Fit will run without any optimizer
    """

    def __call__(self, module: torch.nn.Module):
        raise NotImplementedError


class TorchFactory(Factory):
    """Simple factory wrapping standard PyTorch modules."""

    def __init__(self, module: DictConfig):
        if "_lazy_" in module:
            # A lazy initialization
            module["_target_"] = module["_lazy_"]
            del module["_lazy_"]
        self.module = module

    def __call__(self, **kwargs):
        return hydra.utils.instantiate(self.module, **kwargs)


class OptimizerFactory(TorchFactory):
    def __init__(self, optimizer: DictConfig):
        super().__init__(optimizer)

    def __call__(self, module, **kwargs):
        log.info(f"Instantiating optimizer <{self.module._target_}>")
        params = module.parameters()
        return hydra.utils.instantiate(self.module, params=params, **kwargs)


class WeightDecayOptimizer(Factory, ABC):
    def __init__(self, weight_decay: float, no_decay_params=None):
        if no_decay_params is None:
            no_decay_params = []
        self.weight_decay = weight_decay
        self.no_decay_params = no_decay_params  #

    def group_params(self, module: torch.nn.Module) -> list:
        if self.no_decay_params is not None:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in module.named_parameters() if not any(nd in n for nd in self.no_decay_params)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in module.named_parameters() if any(nd in n for nd in self.no_decay_params)],
                    "weight_decay": 0.0,
                },
            ]

            log.info(
                f'Weight decay will applied to {len(optimizer_grouped_parameters[0]["params"])} and not applied to {len(optimizer_grouped_parameters[1]["params"])} parameters'
            )
        else:

            optimizer_grouped_parameters = [{"params": module.parameters(), "weight_decay": self.weight_decay}]

        return optimizer_grouped_parameters


class OptimizerFactoryWithDecay(TorchFactory):
    def __init__(self, optimizer: DictConfig):
        super().__init__(optimizer)

    def __call__(self, module, **kwargs):
        log.info(f"Instantiating optimizer <{self.module._target_}>")
        weight_decay_param = WeightDecayOptimizer(self.module.weight_decay, self.module.no_decay_params)
        params = weight_decay_param.group_params(module)
        module_conf_with = {
            k: self.module[k]
            for k in self.module
            if k
            not in [
                "no_decay_params",
            ]
        }
        return hydra.utils.instantiate(module_conf_with, params=params, _convert_="partial")
        # return Adam(params, **module_conf_with)


class LRSchedulerFactory:
    def __init__(self, scheduler: DictConfig):
        self.module = SchedulerConfig(scheduler)

    def __call__(self, optimizer, **kwargs):
        log.info(f"Loading learning rate scheduler <{self.module.name}>")
        scheduler = get_scheduler_proxy(config=self.module, optimizer=optimizer)
        return scheduler
