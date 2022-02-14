import os.path

import numpy as np
import pandas as pd
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster

try:
    import rapidjson as json
except:
    import json


class CATHDataset:
    '''
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.

    Has attributes `self.train`, `self.val`, `self.test`, each of which are
    JSON/dictionary-type datasets as described in README.md.

    :param path: path to chain_set.jsonl
    :param splits_path: path to chain_set_splits.json or equivalent.
    '''

    def __init__(self, path, splits_path, top_k=30):
        with open(splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits['train'], \
                                          dataset_splits['validation'], dataset_splits['test']

        self.train, self.val, self.test = [], [], []

        with open(path) as f:
            lines = f.readlines()

        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            name = entry['name']
            coords = entry['coords']

            entry['coords'] = list(zip(
                coords['N'], coords['CA'], coords['C'], coords['O']
            ))

            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)


class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.

    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.

    :param unit_counts: array of node counts in the dataset to sample from
    :param max_units: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''

    def __init__(self, unit_counts, max_units=3000, shuffle=True):
        self.unit_counts = unit_counts
        self.idx = [i for i in range(len(unit_counts))
                    if unit_counts[i] <= max_units]
        self.shuffle = shuffle
        self.max_units = max_units
        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.unit_counts[idx[0]] <= self.max_units:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.unit_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self):
        if not self.batches: self._form_batches()
        return len(self.batches)

    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch

