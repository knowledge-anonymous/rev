import argparse
from typing import List, Optional

# import fix_hook
#
# fix_hook.apply_fixes()
import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

import gbp
from gbp.models.atom_model import Atom3DModel
from gbp.models.cpd_modelv2 import CPDModelv2
from gbp.utils import utils

log = utils.get_logger(__name__)


def test(config: dict) -> Optional[float]:
    if config.get("test", None) is None:
        raise ValueError("Checkpoint path is not defined.")
    if config.get("task", None) is None:
        raise ValueError("Task  is not defined.")
    test = config.get('test')
    task = config.get('task')
    assert task in ['cpd', 'psr', 'lba']
    seed_everything(12345, workers=True)

    # Init lightning datamodule

    if task == 'cpd':
        datamodule = gbp.datamodules.cath_datamodule.CATHDataModule(
            data_dir='./data',
            file_name = "chain_set.jsonl",
            splits_file_name = "chain_set_splits.json",
            short_file_name = "test_split_L100.json",
            single_chain_file_name = "test_split_sc.json",
            max_units = 3000,
            unit = "node",
            num_workers = 12,
            max_neighbors = 30
        )
    else:
        datamodule = gbp.datamodules.atom3d_datamodule.Atom3DDataModule(
            task= task.upper(),
            data_dir='./atom3d-data/',
            max_units = 0,
            edge_cutoff = 4.5,
            num_workers = 12,
            max_neighbors = 32,
            batch_size= 8
        )
    log.info(f"Instantiated datamodule <{datamodule.__class__}>")


    # Init lightning model
    model = torch.load(test)
    # model = Atom3DModel.load_from_checkpoint(checkpoint_path=test)
    # log.info(f"Instantiated model <{model.__class__}>")
    #
    # torch.save(model, f'./models/{task}_model.pt')
    # exit()

    trainer = Trainer(gpus=1, callbacks=None, logger=None, max_epochs=1)
    log.info(f"Instantiated model <{trainer.__class__}>")

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility for running tests.')

    parser.add_argument('model', type=str, default='./models/cpd_model.pt')
    parser.add_argument('task', choices=['cpd', 'psr', 'lba'], default='cpd', help='Task to run')

    args = parser.parse_args()

    test({
        "task": args.task,
        "test": args.model
    })
