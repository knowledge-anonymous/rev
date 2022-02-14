from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Iterable, Dict, Tuple, Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch_scatter
import torchmetrics
import wandb
from atom3d.util import metrics
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import ModuleDict, Tensor
from torchmetrics import MetricCollection, CatMetric
from tqdm import tqdm

from gbp import logger
from gbp.configs.gbp_config import GBPConfig, GBPConvConfig, ModelConfig, GBPConvLayerConfig
from gbp.constants import smp_names, _NUM_ATOM_TYPES
from gbp.datamodules.datasets.protein_graph_dataset import ProteinGraphDataset
from gbp.models.modules.gbp import GBP, GBPConvLayer, LayerNorm, tuple_index, GBPv2
from torch.distributions import Categorical
from torch_scatter import scatter_mean

import torch.nn.functional as F

import sklearn.metrics as sk_metrics
import pytorch_lightning as pl

from gbp.utils import get_activations, dictionary_to_option, str2bool


def get_loss(task):
    return nn.MSELoss()

class Atom3DModel(pl.LightningModule):

    '''
    A base 5-layer GBP-GNN for all ATOM3D tasks, using GBPs with
    vector gating as described in the manuscript. Takes in atomic-level
    structure graphs of type `torch_geometric.data.Batch`
    and returns a single scalar.

    This class should not be used directly. Instead, please use the
    task-specific models which extend BaseLightModel. (Some of these classes
    may be aliases of BaseLightModel.)

    :param num_rbf: number of radial bases to use in the edge embedding
    '''

    def __init__(self,
                 task, smp_idx='0',
                 num_rbf=16, learning_rate=1e-4,
                  dropout=0.1, dense_dropout=0.1, precision=32,

                 loss_fn: Optional[Callable] = None,
                 model: ModelConfig = None,
                 gbp: GBPConfig = None,
                 gbp_conv_layer: GBPConvLayerConfig = None,
                 **kwargs):


        super().__init__()

        self.task = task
        self.smp_idx = smp_idx

        self.label_names = None
        self.loss_means = None
        self.loss_std = None

        self.loss_fn = get_loss(task)
        # self.head = get_prediction_head(self.task)(self)


        self._DEFAULT_V_DIM = (model.vertex_scalar, model.vertex_vector)
        self._DEFAULT_E_DIM = (model.edge_scalar, model.edge_vector)
        # activations = (scalar_act, vector_act)

        #self.save_hyperparameters('num_rbf', 'learning_rate', 'activations', 'encoder_layers', 'scalar_act', 'vector_act')
        self.save_hyperparameters()

        metrics =  self.get_metrics(task).items()
        for phase in ['train', 'valid', 'test']:
            for k, v in metrics:
                setattr(self, 'metric_'+k+'_'+phase, v())

        self.metrics = {phase: nn.ModuleDict(
            {k: getattr(self, 'metric_'+k+'_'+phase) for k, v in metrics}
        ) for phase in ['train', 'valid', 'test']}

        if gbp.vector_gate is False:
            logger.warning('GBP vector gate is False. This is not recommended.')

        self.embed = nn.Embedding(_NUM_ATOM_TYPES, _NUM_ATOM_TYPES)

        self.W_e = nn.Sequential(
            LayerNorm((num_rbf, 1)),
            GBP((num_rbf, 1), self._DEFAULT_E_DIM,
                activations=(None, None), vector_gate=gbp.vector_gate, scalar_gate=gbp.scalar_gate)
        )

        self.W_v = nn.Sequential(
            LayerNorm((_NUM_ATOM_TYPES, 0)),
            GBP((_NUM_ATOM_TYPES, 0), self._DEFAULT_V_DIM,
                activations=(None, None), vector_gate=gbp.vector_gate, scalar_gate=gbp.scalar_gate)
        )

        self.layers = nn.ModuleList(
            GBPConvLayer(self._DEFAULT_V_DIM, self._DEFAULT_E_DIM, drop_rate=dropout, gbp=gbp, gbp_conv_layer=gbp_conv_layer)
            for _ in range(model.encoder_layers))

        ns, _ = self._DEFAULT_V_DIM
        self.W_out = nn.Sequential(
            LayerNorm(self._DEFAULT_V_DIM),
            GBP(self._DEFAULT_V_DIM, (ns, 0),
                activations=gbp.activations, vector_gate=gbp.vector_gate, scalar_gate=gbp.scalar_gate)
        )
        self.W_out_dropout = nn.Dropout(p=self.hparams.dropout)
        self.dense = nn.Sequential(
            nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True),
            nn.Dropout(p=model.dense_dropout),
            nn.Linear(2 * ns, 1)
        )



    def get_label(self, batch):
        if type(batch) in [list, tuple]: batch = batch[0]
        return batch.label

    def _get_num_graphs(self, batch):
        if type(batch) in [list, tuple]: batch = batch[0]
        return batch.num_graphs


    def forward(self, batch, scatter_mean=True, dense=True):
        '''
        Forward pass which can be adjusted based on task formulation.

        :param batch: `torch_geometric.data.Batch` with data attributes
                      as returned from a BaseTransform
        :param scatter_mean: if `True`, returns mean of final node embeddings
                             (for each graph), else, returns embeddings seperately
        :param dense: if `True`, applies final dense layer to reduce embedding
                      to a single scalar; else, returns the embedding
        '''
        #print(batch.atoms.size())
        h_V = self.embed(batch.atoms)
        h_E = (batch.edge_s, batch.edge_v)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        batch_id = batch.batch

        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)
        if scatter_mean: out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        if dense: out = self.dense(out).squeeze(-1)
        return out

    def loop(self, batch, batch_idx, phase="train", dataloader_idx=0):
        label = self.get_label(batch)
        pred = self(batch)

        pred_metric = getattr(self, 'metric_pred_' + phase, None)
        label_metric = getattr(self, 'metric_label_' + phase, None)
        id_metric = getattr(self, 'metric_id_' + phase, None)
        if pred_metric:
            pred_metric.update(pred)
        if label_metric:
            label_metric.update(label)
        if id_metric:
            batch_ids = []
            for id in batch.id:
                batch_ids.append(float(id.replace('T', '-')))
            id_metric.update(batch_ids)

        if self.label_names:
            losses = []
            for i,  name in enumerate(self.label_names):
                label_loss = self.loss_fn(pred[:, i], label[:, i])
                if self.loss_means:
                    label_loss = (label_loss)/(self.loss_std[i]**2)
                losses.append(label_loss)
                self.log(f'{phase}/{name}_loss', label_loss)
            loss = sum(losses)
        else:
            loss = self.loss_fn(pred, label)

            for metric in self.metrics[phase].keys():
                if metric == 'auroc':
                    label = label.type(torch.int)
                pred  = pred.detach()

                self.metrics[phase][metric](pred, label)
                self.log(
                    f"{phase}/" + metric,
                    self.metrics[phase][metric],
                    metric_attribute=self.metrics[phase][metric],
                    on_step=True,
                    on_epoch=True,
                    batch_size=self._get_num_graphs(batch),
                )


        self.log(f"{phase}/loss", loss, batch_size=self._get_num_graphs(batch))

        return loss, pred, label

    def test_step(self, batch, batch_idx):
        loss, pred, label = self.loop(batch, batch_idx, 'test')
        return loss

    def configure_optimizers(self):
        if self.hparams.optim is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        else:
            # print('Using learning rate: ' + str(self.hparams.optim.lr))
            optimizer = self.hparams.optim(module=self)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #                                                        factor=0.9, patience=5,
        #                                                        min_lr=0.00001)
        optimizers = {
            "optimizer": optimizer,
        }
        if "lr_scheduler" in self.hparams and self.hparams.lr_scheduler is not None:
            if callable(self.hparams.lr_scheduler):
                optimizers["lr_scheduler"] = self.hparams.lr_scheduler(optimizer=optimizer)
            else:
                optimizers["lr_scheduler"] = self.hparams.lr_scheduler

        return optimizers


    def get_metrics(self, task):
        correlations = {
            'pearson': torchmetrics.regression.pearson.PearsonCorrCoef,
            'spearman': torchmetrics.regression.spearman.SpearmanCorrCoef,
        }

        current_metric = {
            'PSR': {**correlations},
            'LBA': {**correlations, 'rmse': partial(torchmetrics.regression.mse.MeanSquaredError, squared=False)},
        }[task.split('_')[0]]

        # initialize the metrics
        return current_metric
        # return {k: v() for k, v in current_metric.items()}

    @property
    def custom_metric(self):
        if self.task == 'PSR':
            return True

        return False




