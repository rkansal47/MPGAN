import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .spectral_normalization import SpectralNorm

import logging
from typing import Optional


class LinearNet(nn.Module):
    """
    Module for fully connected networks with leaky relu activations

    Args:
        layers (list): list with layers of the fully connected network,
          optionally containing the input and output sizes inside
          e.g. ``[input_size, ... hidden layers ..., output_size]``
        input_size (list): size of input, if 0 or unspecified, first element of `layers` will be
          treated as the input size
        output_size (list): size of output, if 0 or unspecified, last element of `layers` will be
          treated as the output size
        final_linear (bool): keep the final layer operation linear i.e. no normalization,
          no nonlinear activation.Defaults to False.
        leaky_relu_alpha (float): negative slope of leaky relu. Defaults to 0.2.
        dropout_p (float): dropout fraction after each layer. Defaults to 0.
        batch_norm (bool): use batch norm or not. Defaults to False.
        spectral_norm (bool): use spectral norm or not. Defaults to False.
    """

    def __init__(
        self,
        layers: list,
        input_size: int = 0,
        output_size: int = 0,
        final_linear: bool = False,
        leaky_relu_alpha: float = 0.2,
        dropout_p: float = 0,
        batch_norm: bool = False,
        spectral_norm: bool = False,
    ):
        super(LinearNet, self).__init__()

        self.final_linear = final_linear
        self.leaky_relu_alpha = leaky_relu_alpha
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(p=dropout_p)

        layers = layers.copy()

        if input_size:
            layers.insert(0, input_size)
        if output_size:
            layers.append(output_size)

        self.net = nn.ModuleList()
        if batch_norm:
            self.bn = nn.ModuleList()

        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            self.net.append(linear)
            if batch_norm:
                self.bn.append(nn.BatchNorm1d(layers[i + 1]))

        if spectral_norm:
            for i in range(len(self.net)):
                if i != len(self.net) - 1 or not final_linear:
                    self.net[i] = SpectralNorm(self.net[i])

    def forward(self, x: Tensor):
        """
        Runs input `x` through linear layers and returns output

        Args:
            x (Tensor): input tensor of shape ``[batch size, # input features]``
        """
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i != len(self.net) - 1 or not self.final_linear:
                x = F.leaky_relu(x, negative_slope=self.leaky_relu_alpha)
                if self.batch_norm:
                    x = self.bn[i](x)
            x = self.dropout(x)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(net = {self.net})"


# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class MAB(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_layers: list = [],
        layer_norm: bool = False,
        dropout_p: float = 0.0,
        final_linear: bool = True,
        linear_args={},
    ):
        super(MAB, self).__init__()

        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = LinearNet(
            ff_layers,
            input_size=embed_dim,
            output_size=embed_dim,
            final_linear=final_linear,
            **linear_args,
        )

        self.layer_norm = layer_norm

        if self.layer_norm:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor, y: Tensor, y_mask: Tensor = None):
        if y_mask is not None:
            # torch.nn.MultiheadAttention needs a mask of shape [batch_size * num_heads, N, N]
            y_mask = torch.repeat_interleave(y_mask, self.num_heads, dim=0)

        x = x + self.attention(x, y, y, attn_mask=y_mask, need_weights=False)[0]
        if self.layer_norm:
            x = self.norm1(x)
        x = self.dropout(x)

        x = x + self.ff(x)
        if self.layer_norm:
            x = self.norm2(x)
        x = self.dropout(x)

        return x


# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class SAB(nn.Module):
    def __init__(self, **mab_args):
        super(SAB, self).__init__()
        self.mab = MAB(**mab_args)

    def forward(self, x: Tensor, mask: Tensor = None):
        if mask is not None:
            # torch.nn.MultiheadAttention needs a mask vector for each target node
            # i.e. reshaping from [B, N, 1] -> [B, N, N]
            mask = mask.transpose(-2, -1).repeat((1, mask.shape[-2], 1))

        return self.mab(x, x, mask)


# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class PMA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_seeds: int,
        **mab_args,
    ):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, embed_dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(embed_dim, **mab_args)

    def forward(self, x: Tensor, mask: Tensor = None):
        if mask is not None:
            mask = mask.transpose(-2, -1)

        return self.mab(self.S.repeat(x.size(0), 1, 1), x, mask)


# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class ISAB(nn.Module):
    def __init__(self, num_inds, embed_dim, **mab_args):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, embed_dim))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(embed_dim=embed_dim, **mab_args)
        self.mab1 = MAB(embed_dim=embed_dim, **mab_args)

    def forward(self, X, num_inds, mask: Tensor = None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        if mask is not None:
            mask = mask.tanspose(-2,-1).repeat((1,mask.shape[-2], 1))
            mask = torch.split(mask,num_inds,2)[0].shape
        return self.mab1(X, H, mask)


def _attn_mask(mask: Tensor) -> Optional[Tensor]:
    """
    Convert JetNet mask scheme (1 - real, 0 -padded) to nn.MultiHeadAttention mask scheme
    (True - ignore, False - attend)
    """
    if mask is None:
        return None
    else:
        return (1 - mask).bool()


class GAPT_G(nn.Module):
    def __init__(
        self,
        num_particles: int,
        output_feat_size: int,
        sab_layers: int = 2,
        num_heads: int = 4,
        embed_dim: int = 32,
        sab_fc_layers: list = [],
        layer_norm: bool = False,
        dropout_p: float = 0.0,
        final_fc_layers: list = [],
        use_mask: bool = True,
        use_isab: bool = False,
        num_isab_nodes: int = 10,
        linear_args: dict = {},
    ):
        super(GAPT_G, self).__init__()
        self.num_particles = num_particles
        self.output_feat_size = output_feat_size
        self.use_mask = use_mask

        self.sabs = nn.ModuleList()

        sab_args = {
            "embed_dim": embed_dim,
            "ff_layers": sab_fc_layers,
            "final_linear": False,
            "num_heads": num_heads,
            "layer_norm": layer_norm,
            "dropout_p": dropout_p,
            "linear_args": linear_args,
        }

        # intermediate layers
        for _ in range(sab_layers):
            self.sabs.append(SAB(**sab_args) if not use_isab else ISAB(num_isab_nodes, **sab_args))

        self.final_fc = LinearNet(
            final_fc_layers,
            input_size=embed_dim,
            output_size=output_feat_size,
            final_linear=True,
            **linear_args,
        )

    def forward(self, x: Tensor, labels: Tensor = None):
        if self.use_mask:
            # unnormalize the last jet label - the normalized # of particles per jet
            # (between 1/``num_particles`` and 1) - to between 0 and ``num_particles`` - 1
            num_jet_particles = (labels[:, -1] * self.num_particles).int() - 1
            # sort the particles bythe first noise feature per particle, and the first
            # ``num_jet_particles`` particles receive a 1-mask, the rest 0.
            mask = (
                (x[:, :, 0].argsort(1).argsort(1) <= num_jet_particles.unsqueeze(1))
                .unsqueeze(2)
                .float()
            )
            logging.debug(
                f"x \n {x[:2, :, 0]} \n num particles \n {num_jet_particles[:2]} \n gen mask \n {mask[:2]}"
            )
        else:
            mask = None

        for sab in self.sabs:
            x = sab(x, _attn_mask(mask))

        x = torch.tanh(self.final_fc(x))

        return torch.cat((x, mask - 0.5), dim=2) if mask is not None else x


class GAPT_D(nn.Module):
    def __init__(
        self,
        num_particles: int,
        input_feat_size: int,
        sab_layers: int = 2,
        num_heads: int = 4,
        embed_dim: int = 32,
        sab_fc_layers: list = [],
        layer_norm: bool = False,
        dropout_p: float = 0.0,
        final_fc_layers: list = [],
        use_mask: bool = True,
        use_isab: bool = False,
        num_isab_nodes: int = 10,
        linear_args: dict = {},
    ):
        super(GAPT_D, self).__init__()
        self.num_particles = num_particles
        self.input_feat_size = input_feat_size
        self.use_mask = use_mask

        self.sabs = nn.ModuleList()

        sab_args = {
            "embed_dim": embed_dim,
            "ff_layers": sab_fc_layers,
            "final_linear": False,
            "num_heads": num_heads,
            "layer_norm": layer_norm,
            "dropout_p": dropout_p,
            "linear_args": linear_args,
        }

        self.input_embedding = LinearNet(
            [], input_size=input_feat_size, output_size=embed_dim, **linear_args
        )

        # intermediate layers
        for _ in range(sab_layers):
            self.sabs.append(SAB(**sab_args) if not use_isab else ISAB(num_isab_nodes, **sab_args))

        self.pma = PMA(
            num_seeds=1,
            **sab_args,
        )

        self.final_fc = LinearNet(
            final_fc_layers,
            input_size=embed_dim,
            output_size=1,
            final_linear=True,
            **linear_args,
        )

    def forward(self, x: Tensor, labels: Tensor = None):
        if self.use_mask:
            mask = x[..., -1:] + 0.5
            x = x[..., :-1]
        else:
            mask = None

        x = self.input_embedding(x)

        for sab in self.sabs:
            x = sab(x, _attn_mask(mask))

        return torch.sigmoid(self.final_fc(self.pma(x, _attn_mask(mask)).squeeze()))
