from copy import copy
from typing import Callable, Union, Optional, Tuple

import pytorch_lightning
import torch, functools
from dataclasses import dataclass
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from gbp import logger
from gbp.configs.gbp_config import GBPConvLayerConfig
from gbp.configs.gbp_config import GBPConfig, GBPConvConfig


def tuple_sum(*args):
    """
    Sums any number of tuples (s, V) elementwise.
    """
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    """
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    """
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    """
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    """
    return x[0][idx], x[1][idx]


def randn(n, dims, device="cpu"):
    """
    Returns random tuples (s, V) drawn elementwise from a normal distribution.

    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)

    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    """
    return torch.randn(n, dims[0], device=device), torch.randn(n, dims[1], 3, device=device)


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    """
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    """
    v = torch.reshape(x[..., -3 * nv :], x.shape[:-1] + (nv, 3))
    s = x[..., : -3 * nv]
    return s, v


def _merge(s, v):
    """
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    """
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)


def nan_to_identity(activation):
    if activation is None:
        return nn.Identity()
    return activation


def is_identity(activation):
    return activation is None or isinstance(activation, nn.Identity)


class GBP(nn.Module):
    """
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims,
        out_dims,
        h_dim=None,
        activations: Tuple[Optional[Callable]] = (F.relu, torch.sigmoid),
        scalar_gate_act: Optional[Callable] = None,
        vector_gate: bool = False,
        scalar_gate: int = 0,
        bottleneck: int = 1,
        vector_residual=False,
    ):
        super(GBP, self).__init__()
        self.vector_residual = vector_residual
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        self.scalar_gate = scalar_gate
        self.scalar_gate_act = scalar_gate_act if scalar_gate_act else nn.Identity()
        if self.vi:
            if bottleneck == 1:
                self.h_dim = h_dim or max(self.vi, self.vo)
            else:
                assert (
                    self.vi % bottleneck == 0
                ), f"Input channel of vector ({self.vi}) must be divisible with bottleneck factor ({bottleneck})"
                self.h_dim = self.vi // bottleneck
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)


            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.so)

        self.scalar_act, self.vector_act = nan_to_identity(activations[0]), nan_to_identity(activations[1])
        self.dummy_param = nn.Parameter(torch.empty(0))

    def scalar_gate_module(self, s):
        s_gate = None
        return s, s_gate

    def gen_dummy_param(self, n):
        return torch.zeros(n, self.vo, 3, device=self.dummy_param.device)

    def forward(self, x):

        if self.vi:
            s, v = x
            v_pre = torch.transpose(v, -1, -2)

            vh = self.wh(v_pre)
            vn = _norm_no_nan(vh, axis=-2)
            merged = torch.cat([s, vn], -1)
        else:
            merged = x

        s = self.ws(merged)
        s, s_gate = self.scalar_gate_module(s)

        if self.vi and self.vo:
            v = self.wv(vh)
            if self.vector_residual:
                v += v_pre
            v = torch.transpose(v, -1, -2)
            if self.vector_gate:
                gate = self.wsv(self.vector_act(s))
                v = v * torch.sigmoid(gate).unsqueeze(-1)
            elif not is_identity(self.vector_act):
                v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))


        s = self.scalar_act(s)
        v = self.gen_dummy_param(s.shape[0]) if self.vo and not self.vi else v
        return (s, v) if self.vo else s


def GBPv2(in_dims, out_dims, h_dim=None, gbp: GBPConfig = None, **kwargs):
    """
    Wrapper around GBP that takes GBPConfig as input.

    """
    gbp_dict = copy(gbp.__dict__)
    gbp_dict["activations"] = gbp.activations
    del gbp_dict["scalar_act"]
    del gbp_dict["vector_act"]

    # Override with kwargs, if provided
    for key in kwargs:
        gbp_dict[key] = kwargs[key]

    return GBP(in_dims, out_dims, h_dim, **gbp_dict)


class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    """

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    """
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn



class GBPConv(MessagePassing):
    """
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GBPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GBPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GBPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims,
        out_dims,
        edge_dims,
        module_list=None,
        aggr="mean",
        gbp: GBPConfig = None,
        gbp_conv: GBPConvConfig = None,
        **kwargs,
    ):
        super(GBPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        self.self_message = gbp_conv.self_message

        if gbp_conv.self_message:
            in_scalar = 2 * self.si + self.se
            in_vector = 2 * self.vi + self.ve
        else:
            in_scalar = self.si + self.se
            in_vector = self.vi + self.ve

        soft_gbp = gbp.duplicate(bottleneck=GBPConfig.bottleneck, vector_residual=GBPConfig.vector_residual)
        GBP_ = functools.partial(GBPv2, gbp=soft_gbp)
        GBP__ = functools.partial(GBPv2, gbp=gbp)

        self.gbp_conv = gbp_conv


        module_list = module_list or []
        if not module_list:
            if gbp_conv.n_message == 1:
                module_list.append(
                    GBP_(
                        (in_scalar, in_vector),
                        (self.so, self.vo),
                        activations=(None, None),
                    )
                )
            else:
                hidden_dims = (
                    out_dims[0] * gbp_conv.message_ff_multiplier,
                    out_dims[1] * gbp_conv.message_ff_multiplier,
                )
                module_list.append(
                    GBP_((in_scalar, in_vector), hidden_dims)
                    # GBP_((2*self.si , 2*self.vi), out_dims) # no edge
                )

                for i in range(gbp_conv.n_message - 2):
                    module_list.append(GBP__(hidden_dims, hidden_dims))

                module_list.append(GBP_(hidden_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        """
        x_s, x_v = x

        message = self.propagate(
            edge_index,
            s=x_s,
            v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]),
            edge_attr=edge_attr,
        )
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        # print( self.vi self.ve)
        # print(v_j.size(), edge_attr[1].size(), v_i.size())
        # print(self.message_func)
        # print(message[1].size())
        message = self.message_func(message)
        return _merge(*message)


class GBPConvLayer(nn.Module):
    """
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GBPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GBPs to use in message function
    :param n_feedforward: number of GBPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GBPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GBPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        node_dims,
        edge_dims,
        n_feedforward=2,
        drop_rate=0.1,
        autoregressive=False,
        conv_type="GBP",
        ff_activations=None,
        gbp: GBPConfig = None,
        gbp_conv_layer: GBPConvLayerConfig = None,
        **kwargs,
    ):

        super(GBPConvLayer, self).__init__()

        conv_class = GBPConv

        if ff_activations is None:
            ff_activations = gbp.activations

        self.pre_norm = gbp_conv_layer.pre_norm
        self.conv = conv_class(
            node_dims,
            node_dims,
            edge_dims,
            aggr="add" if autoregressive else "mean",
            gbp=gbp,
            gbp_conv=gbp_conv_layer.gbp_conv,
        )

        GBP_ = functools.partial(GBPv2, gbp=gbp)
        ff_gbp = gbp.duplicate()
        ff_gbp.activations = ff_activations

        ff_gbp_fixed = gbp.duplicate(vector_residual=False)

        GBP_ff = functools.partial(GBPv2, gbp=ff_gbp)
        GBP_ff_fix = functools.partial(GBPv2, gbp=ff_gbp_fixed)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if gbp_conv_layer.n_feedforward == 1:
            ff_func.append(GBP_ff_fix(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GBP_ff_fix(node_dims, hid_dims))
            for i in range(gbp_conv_layer.n_feedforward - 2):
                ff_func.append(GBP_ff(hid_dims, hid_dims))
            ff_func.append(GBP_ff_fix(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node
                embeddings `x` will still be the base of the update and the
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        """

        if self.pre_norm:
            x = self.norm[0](x)

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward),
            )

            count = scatter_add(torch.ones_like(dst), dst, dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = tuple_sum(x, self.dropout[0](dh))

        if self.pre_norm:
            x = self.norm[1](x)
        else:
            x = self.norm[0](x)

        dh = self.ff_func(x)
        x = tuple_sum(x, self.dropout[1](dh))

        if not self.pre_norm:
            x = self.norm[1](x)

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x
