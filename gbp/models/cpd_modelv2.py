from functools import partial
from typing import Callable, Optional, Iterable, Dict, Tuple, Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import wandb
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import ModuleDict, Tensor
from tqdm import tqdm

from gbp import logger
from gbp.configs.gbp_config import  GBPConvConfig, ModelConfig, GBPConvLayerConfig
from gbp.configs.gbp_config import GBPConfig
from gbp.datamodules.datasets.protein_graph_dataset import ProteinGraphDataset
from gbp.models.modules.gbp import GBP, GBPConvLayer, LayerNorm, tuple_index, GBPv2
from torch.distributions import Categorical
from torch_scatter import scatter_mean

import torch.nn.functional as F

import pytorch_lightning as pl

from gbp.utils import get_activations, dictionary_to_option, str2bool


class CPDModelv2(pl.LightningModule):
    """Contains logic layer for GBP based encoder/decoder models."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
        dense_dropout: float = 0.1,
        precision: int = 32,
        loss_fn: Optional[Callable] = None,
        metrics: Optional[ModuleDict] = None,
        model: ModelConfig = None,
        gbp: GBPConfig = None,
        gbp_conv_layer: GBPConvLayerConfig = None,
        log_perplexity: bool = True,
        test_recovery: bool = True,
        **kwargs,
    ):
        """
        Args:
            decoder: Type of decoder to be used for GBP-GNN
            node_in_dim: Dimensions of node inputs
            edge_in_dim: Dimensions of edge inputs

        """
        super().__init__()



        self.log_perplexity = log_perplexity

        # To handle subset tests
        self.test_recovery = test_recovery
        self.recovery_metrics = []
        self.subsets = {}

        if model is None:
            raise ValueError("Model configuration not provided")

        if gbp is None:
            raise ValueError("GBP configuration not provided")


        if gbp_conv_layer is None:
            raise ValueError("GBP-Conv configuration not provided")

        self.loss_fn = loss_fn
        if metrics is None:
            metrics = {}
        self.metrics = metrics

        self.save_hyperparameters()

        self._DEFAULT_V_DIM = (model.vertex_scalar, model.vertex_vector)
        self._DEFAULT_E_DIM = (model.edge_scalar, model.edge_vector)

        self.loss_fn = nn.CrossEntropyLoss()

        # TODO find a efficient way to pass the hyper parameters
        # , scalar_only = gbp.scalar_only
        self.W_v = nn.Sequential(
            GBP(node_in_dim, self._DEFAULT_V_DIM, activations=(None, None)),
            LayerNorm(self._DEFAULT_V_DIM)
        )
        self.W_e = nn.Sequential(
            GBP(edge_in_dim, self._DEFAULT_E_DIM, activations=(None, None)),
            LayerNorm(self._DEFAULT_E_DIM)
        )

        self.encoder_layers = nn.ModuleList(
            GBPConvLayer(self._DEFAULT_V_DIM, self._DEFAULT_E_DIM, drop_rate=dropout, gbp=gbp, gbp_conv_layer=gbp_conv_layer)
            for _ in range(model.encoder_layers))

        self.W_s = nn.Embedding(20, 20)
        edge_h_dim = (self._DEFAULT_E_DIM[0] + 20, self._DEFAULT_E_DIM[1])

        self.decoder_layers = nn.ModuleList(
            GBPConvLayer(self._DEFAULT_V_DIM, edge_h_dim,
                         drop_rate=dropout, autoregressive=True,
                         gbp=gbp, gbp_conv_layer=gbp_conv_layer)
            for _ in range(model.decoder_layers))

        self.W_out = GBP(self._DEFAULT_V_DIM, (20, 0), activations=(None, None))


    def forward(self, batch):
        '''
        Forward pass to be used at train-time, or evaluating likelihood.

        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: int `torch.Tensor` of shape [num_nodes]
        '''


        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        encoder_embeddings = h_V

        h_S = self.W_s(batch.seq)
        h_S = h_S[batch.edge_index[0]]
        h_S[batch.edge_index[0] >= batch.edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])

        for layer in self.decoder_layers:
            h_V = layer(h_V, batch.edge_index, h_E, autoregressive_x=encoder_embeddings)

        logits = self.W_out(h_V)

        return logits

    def sample(self, h_V, edge_index, h_E, n_samples, temperature=0.1):
        '''
        Samples sequences autoregressively from the distribution
        learned by the model.

        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param n_samples: number of samples
        :param temperature: temperature to use in softmax
                            over the categorical distribution

        :return: int `torch.Tensor` of shape [n_samples, n_nodes] based on the
                 residue-to-int mapping of the original training data
        '''

        with torch.no_grad():

            device = edge_index.device
            L = h_V[0].shape[0]

            h_V = self.W_v(h_V)
            h_E = self.W_e(h_E)

            for layer in self.encoder_layers:
                h_V = layer(h_V, edge_index, h_E)

            h_V = (h_V[0].repeat(n_samples, 1),
                   h_V[1].repeat(n_samples, 1, 1))

            h_E = (h_E[0].repeat(n_samples, 1),
                   h_E[1].repeat(n_samples, 1, 1))

            edge_index = edge_index.expand(n_samples, -1, -1)
            offset = L * torch.arange(n_samples, device=device).view(-1, 1, 1)
            edge_index = torch.cat(tuple(edge_index + offset), dim=-1)

            seq = torch.zeros(n_samples * L, device=device, dtype=torch.int)
            h_S = torch.zeros(n_samples * L, 20, device=device)

            h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]

            for i in range(L):

                h_S_ = h_S[edge_index[0]]
                h_S_[edge_index[0] >= edge_index[1]] = 0
                h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])

                edge_mask = edge_index[1] % L == i
                edge_index_ = edge_index[:, edge_mask]
                h_E_ = tuple_index(h_E_, edge_mask)
                node_mask = torch.zeros(n_samples * L, device=device, dtype=torch.bool)
                node_mask[i::L] = True

                for j, layer in enumerate(self.decoder_layers):
                    out = layer(h_V_cache[j], edge_index_, h_E_,
                                autoregressive_x=h_V_cache[0], node_mask=node_mask)

                    out = tuple_index(out, node_mask)

                    if j < len(self.decoder_layers) - 1:
                        h_V_cache[j + 1][0][i::L] = out[0]
                        h_V_cache[j + 1][1][i::L] = out[1]

                logits = self.W_out(out)
                seq[i::L] = Categorical(logits=logits / temperature).sample()
                h_S[i::L] = self.W_s(seq[i::L])

            return seq.view(n_samples, L)


    def recovery_from_protein(self, protein):
        h_V = (protein.node_s, protein.node_v)
        h_E = (protein.edge_s, protein.edge_v)
        sample = self.sample(h_V, protein.edge_index,
                             h_E, n_samples=100)
        recovery_ = sample.eq(protein.seq).float().mean()
        return recovery_

    def recovery_test_samples(self, samples):
        for protein in samples:
            recovery_ = self.recovery_from_protein(protein)

            print(protein.name, recovery_, flush=True)

            self.recovery_metrics['all'].update(recovery_)
            for subset in self.subsets:
                if protein.name in self.subsets[subset]:
                    self.recovery_metrics[subset].update(recovery_)

    def load_splits(self):
        if getattr(self.trainer.datamodule, 'custom_splits', None) is not None:
            self.subsets = self.trainer.datamodule.custom_splits
            logger.info(f"Using custom splits: {', '.join(list(self.subsets.keys()))}")

    def on_test_start(self):
        self.load_splits()
        metric_keys = list(self.subsets.keys()) + ['all']
        self.recovery_metrics = nn.ModuleDict(
            {key: torchmetrics.CatMetric() for key in metric_keys}
        )

    def calculate_recovery_metrics(self):
        metric_keys = list(self.subsets.keys()) + ['all']
        output = {}
        for key in metric_keys:
            out = self.recovery_metrics[key].compute()
            recovery = torch.median(torch.tensor(out))
            output[key] = recovery
        return output

    def on_test_epoch_end(self):
        super().on_test_epoch_end()

        output = self.calculate_recovery_metrics()
        for key in output:
            self.log(f"test/recovery/"+key, output[key])
            print(f"test/recovery/"+key, output[key].item(), flush=True)

    def recovery(self, datamodule: LightningDataModule):
        self.on_test_start()
        self.eval()
        dataset = datamodule.testset
        self.recovery_test_samples(dataset)
        data = self.calculate_recovery_metrics()
        print(f'TEST recovery: {data}')

    def loop(self, batch, batch_idx, phase="train", dataloader_idx=0):
        # label = self.get_label(batch)

        logits = self(batch)
        logits, label = logits[batch.mask], batch.seq[batch.mask]
        loss = self.loss_fn(logits, label)

        # TODO scale loss by number of nodes
        self.log(f"{phase}/loss", loss, batch_size=batch.num_graphs)
        if self.log_perplexity:
            self.log(f"{phase}/perplexity", torch.exp(loss), batch_size=batch.num_graphs)
        if phase == 'test' and dataloader_idx == 0 and self.test_recovery:
            self.recovery_test_samples(batch.to_data_list())

        for metric in self.metrics:
            pred_metric = logits
            if metric == "auroc":
                label = label.type(torch.int)

            self.metrics[metric](pred_metric, label)
            self.log(
                f"{phase}/" + metric,
                self.metrics[metric],
                metric_attribute=self.metrics[metric],
                on_step=True,
                on_epoch=True,
                batch_size=batch.num_graphs,
            )

        return loss, logits, label


    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, pred, label = self.loop(batch, batch_idx, "test", dataloader_idx=dataloader_idx)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, pred, label = self.loop(batch, batch_idx, "test", dataloader_idx=dataloader_idx)

        return loss

    def configure_optimizers(self):
        if self.hparams.optim is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        else:
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
