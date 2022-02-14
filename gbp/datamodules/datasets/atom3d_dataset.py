import traceback
from functools import partial

import pandas
import torch, random, scipy, math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from atom3d.datasets import LMDBDataset

import gbp.neighbors as nb
from torch.utils.data import IterableDataset

from gbp import logger
from gbp.constants import _NUM_ATOM_TYPES, _element_mapping, _amino_acids, smp_names
from gbp.models.modules.gbp import GBP, GBPConvLayer, LayerNorm
import torch_cluster, torch_geometric, torch_scatter


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


# value = 1.3
# D_mu, D_sigma, rbfed = _rbf(torch.tensor([value]), 0, 20, 4)
# back = (torch.sqrt(-torch.log(rbfed[0])) * D_sigma)[0]
def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    # -torch.log(RBF)
    return RBF


def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1),
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num,
                         (edge_s, edge_v))

    return edge_s, edge_v



class BaseTransform:
    '''
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs. This class should not be used directly; instead, use the
    task-specific transforms, which all extend BaseTransform. Node
    and edge features are as described in the EGNN manuscript.

    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -atoms      numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]

    Subclasses of BaseTransform will produce graphs with additional
    attributes for the tasks-specific training labels, in addition
    to the above.

    All subclasses of BaseTransform directly inherit the BaseTransform
    constructor.

    :param edge_cutoff: distance cutoff to use when drawing edges
    :param num_rbf: number of radial bases to encode the distance on each edge
    :device: if "cuda", will do preprocessing on the GPU
    '''

    def __init__(self, edge_cutoff=4.5, num_rbf=16, max_num_neighbors=64, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, df, edge_index=None):
        '''
        :param df: `pandas.DataFrame` of atomic coordinates
                    in the ATOM3D format

        :return: `torch_geometric.data.Data` structure graph
        '''
        with torch.no_grad():
            coords = df[['x', 'y', 'z']].to_numpy()
            coords = torch.as_tensor(coords,
                                     dtype=torch.float32, device=self.device)
            atoms = torch.as_tensor(list(map(_element_mapping, df.element)),
                                    dtype=torch.long, device=self.device)

            edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff,
                                                    max_num_neighbors=self.max_num_neighbors)

            edge_s, edge_v = _edge_features(coords, edge_index,
                                            D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)

            return torch_geometric.data.Data(x=coords, atoms=atoms,
                        edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)


########################################################################


class LBATransform(BaseTransform):
    '''
    Transforms dict-style entries from the ATOM3D LBA dataset
    to featurized graphs. Returns a `torch_geometric.data.Data`
    graph with attribute `label` for the neglog-affinity
    and all structural attributes as described in BaseTransform.

    The transform combines the atomic coordinates of the pocket
    and ligand atoms and treats them as a single structure / graph.

    Includes hydrogen atoms.
    '''

    def __call__(self, elem, index=-1):
        pocket, ligand = elem['atoms_pocket'], elem['atoms_ligand']
        df = pd.concat([pocket, ligand], ignore_index=True)

        data = super().__call__(df)
        with torch.no_grad():
            data.label = elem['scores']['neglog_aff']
            lig_flag = torch.zeros(df.shape[0], device=self.device, dtype=torch.bool)
            lig_flag[-len(ligand):] = 1
            data.lig_flag = lig_flag
        return data



class PSRTransform(BaseTransform):
    '''
    Transforms dict-style entries from the ATOM3D PSR dataset
    to featurized graphs. Returns a `torch_geometric.data.Data`
    graph with attribute `label` for the GDT_TS, `id` for the
    name of the target, and all structural attributes as
    described in BaseTransform.

    Includes hydrogen atoms.
    '''

    def __call__(self, elem, index=-1):
        df = elem['atoms']
        df = df[df.element != 'H'].reset_index(drop=True)
        data = super().__call__(df, elem.get('edge_index', None))
        data.label = elem['scores']['gdt_ts']
        data.id = eval(elem['id'])[0]
        return data

