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

import gzip
import io
import os.path
import shutil
import traceback
from functools import partial
from multiprocessing import Pool
from typing import Optional

import atom3d.datasets
import lmdb
import pandas as pd
import torch
import torch_geometric
from atom3d.datasets.datasets import deserialize, LMDBDataset, make_lmdb_dataset, serialize
from tqdm import tqdm

import gbp
from gbp.datamodules.sampler import DistributedSamplerWrapper, BatchSampler
from .datasets import atom3d_dataset 
import pytorch_lightning as pl

import logging
log = logger = logging.getLogger('gbp')

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def get_data_path(dataset, lba_split=None):
    data_paths = {
        'PSR': 'PSR/splits/split-by-year/data/',
        'LBA': f'LBA/splits/split-by-sequence-identity-{lba_split}/data/',
    }

    if dataset not in data_paths:
        raise NotImplementedError(f'Dataset {dataset} is not implemented yet, please choose one of the following datasets: '
                                  f'{", ".join(list(data_paths.keys()))}')

    return data_paths[dataset]

def get_task_split(task, lba_split=None):
    splits = {
        'PSR': 'year',
        'LBA': f'sequence-identity-{lba_split}',
    }

    if task not in splits:
        raise NotImplementedError(f'Dataset {task} is not implemented yet, please choose one of the following datasets: '
                                  f'{", ".join(list(splits.keys()))}')
    return splits[task]



class Atom3DDataModule(pl.LightningDataModule):
    """
    A data wrapper for Atom3D package. It downloads the missing
    data files from Zenodo. Applies the transformations to the
    raw data to gather invariant features.

    :param task: name of the task
    :param lba_split: data split for LBA task (30 or 60).
    :param data_dir: location of where the data is stored for the tasks.
    :param batch_size: mini-batch size
    :param num_workers:  number of workers to be used for data loading
    :param edge_cutoff: distance threshold value to determine the edges and RBF kernel.
    :param max_neighbors: number of maximum neighbors for a given node
    :param in_memory: copies the dataset to RAM for faster loading times
    :param clean_up: if `in_memory` enabled, cleans the memory after the run
    :param preprocess: if enabled, transform the raw data and stores in pickle object
    to elevate the burden from CPU.
    """
    def __init__(self, task: str, lba_split: Optional[int] = 30,
                 data_dir: str = "atom3d-data/", batch_size: int = 32, num_workers: int =4,
                 edge_cutoff: float =4.5, max_neighbors: int =32, in_memory: bool =False,
                 clean_up: bool =False, preprocess: bool =True, max_units: int = 0,
                  unit="edge", **kwargs):
        super().__init__()
        self.max_units = max_units
        self.unit = unit
        self.task = task
        self.lba_split = lba_split
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.num_workers = num_workers
        self.clean_up = clean_up
        self.in_memory = in_memory
        self.max_neighbors = max_neighbors
        self.edge_cutoff = edge_cutoff

        self.memory_path =  '/dev/shm/'

        self.transformations = {
                'PSR': atom3d_dataset.PSRTransform,
                'LBA': atom3d_dataset.LBATransform,
        }


    def get_cache_params(self):
        return f'e{self.edge_cutoff}-n{self.max_neighbors}'

    def prepare_data(self):
        # download
        relative_path = get_data_path(self.task, self.lba_split)
        full_path = os.path.join(self.data_dir, relative_path)
        if not os.path.exists(full_path):
            log.info('Downloading the dataset...')
            atom3d.datasets.download_dataset(self.task.split('_')[0], split=get_task_split(self.task, self.lba_split),
                out_path=os.path.join(self.data_dir, os.sep.join(relative_path.split('/')[:2]))
            )


    def setup(self, stage: Optional[str] = None):
        # check if exists
        relative_path = get_data_path(self.task, self.lba_split)
        full_path = os.path.join(self.data_dir, relative_path)
        if self.in_memory:
            # Move files to ram-disk
            full_path_new = os.path.join(self.memory_path, relative_path)
            if not os.path.exists(full_path_new):
                full_path = full_path_new

        if stage is None:
            need = set(['train', 'valid', 'test'])
        elif stage == 'fit':
            need = set(['train', 'valid'])
        elif stage == 'validate':
            need = set(['valid'])
        elif stage == 'test':
            need = set(['test'])


        transform = self.transformations[self.task](edge_cutoff=self.edge_cutoff, max_num_neighbors=self.max_neighbors)
        print(f'Number of neighbors {self.max_neighbors}')
        print(f'Edge cutoff {self.edge_cutoff}')
        dataset_class = partial(LMDBDataset, transform=transform)
        file_name_postfix = ''

        if 'train' in need:
            trainset = dataset_class(full_path + 'train' +file_name_postfix)
        if 'valid' in need:
            valset = dataset_class(full_path + 'val'+file_name_postfix)
        if 'test' in need:
            testset = dataset_class(full_path + 'test'+file_name_postfix)

        if 'train' in need:
            self.train_set = trainset
        if 'valid' in need:
            self.val_set = valset
        if 'test' in need:
            self.test_set = testset


    def get_dataloader(self, data, batch_size=None, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size
        log.debug(f'Initializing data loader with batch size {batch_size}')
        if self.max_units == 0:
            dataloader = partial(torch_geometric.loader.DataLoader,
                             num_workers=self.num_workers, batch_size=batch_size,
                             prefetch_factor=100, worker_init_fn=set_worker_sharing_strategy)
        else:
            if torch.distributed.is_initialized():
                dataloader = lambda x: torch_geometric.loader.DataLoader(
                    x,
                    num_workers=self.num_workers,
                    batch_sampler=DistributedSamplerWrapper(
                        BatchSampler(getattr(x, self.unit + "_counts"), max_units=self.max_units, shuffle=shuffle)
                    ),
                    pin_memory=True
                    # multiprocessing_context="fork",
                )
            else:
                dataloader = lambda x: torch_geometric.loader.DataLoader(
                    x,
                    num_workers=self.num_workers,
                    batch_sampler=BatchSampler(
                        getattr(x, self.unit + "_counts"), max_units=self.max_units, shuffle=shuffle
                    ),
                    pin_memory=True
                    # multiprocessing_context="fork",
                )
            return dataloader(data)

        #if self.task not in ['PPI', 'RES']:
        dataloader = partial(dataloader, shuffle=shuffle)

        return dataloader(data)

    def train_dataloader(self):
        return self.get_dataloader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage: Optional[str] = None):
        # Clean up the memory disk, might cause problems for concurrent runs.
        if self.in_memory and self.clean_up:
            relative_path = get_data_path(self.task, self.lba_split)
            full_path = os.path.join(self.memory_path, relative_path)
            shutil.rmtree(full_path)


def get_datasets(task, lba_split=30, max_num_neighbors=32, edge_cutoff=4.5):
    data_path = {
        'PSR': 'PSR/splits/split-by-year/data/',
        'LBA': f'LBA/splits/split-by-sequence-identity-{lba_split}/data/',
    }[task]

    if task == 'RES':
        split_path = 'RES/splits/split-by-cath-topology/indices/'
        dataset = partial(atom3d_dataset.RESDataset, data_path)
        trainset = dataset(split_path=split_path + 'train_indices.txt')
        valset = dataset(split_path=split_path + 'val_indices.txt')
        testset = dataset(split_path=split_path + 'test_indices.txt')

    elif task == 'PPI':
        trainset = atom3d_dataset.PPIDataset(data_path + 'train')
        valset = atom3d_dataset.PPIDataset(data_path + 'val')
        testset = atom3d_dataset.PPIDataset(data_path + 'test')

    else:
        transform = {
            'PSR': atom3d_dataset.PSRTransform,
            'LBA': atom3d_dataset.LBATransform,
        }[task](edge_cutoff=edge_cutoff, max_num_neighbors=max_num_neighbors, )
        print(f'Number of neighbors {max_num_neighbors}')
        print(f'Edge cutoff {edge_cutoff}')

        trainset = LMDBDataset(data_path + 'train', transform=transform)
        valset = LMDBDataset(data_path + 'val', transform=transform)
        testset = LMDBDataset(data_path + 'test', transform=transform)

    return trainset, valset, testset
