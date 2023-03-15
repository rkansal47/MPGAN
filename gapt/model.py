import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from .spectral_normalization import SpectralNorm
import math
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
        layer_norm: bool = False,
        spectral_norm: bool = False,
    ):
        super(LinearNet, self).__init__()
        self.final_linear = final_linear
        self.leaky_relu_alpha = leaky_relu_alpha
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout = nn.Dropout(p=dropout_p)

        layers = layers.copy()

        if input_size:
            layers.insert(0, input_size)
        if output_size:
            layers.append(output_size)

        self.net = nn.ModuleList()
        if batch_norm:
            self.bn = nn.ModuleList()
        if layer_norm:
            self.ln = nn.ModuleList()

        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            self.net.append(linear)

            if batch_norm:
                self.bn.append(nn.BatchNorm1d(layers[i + 1]))
            
            if layer_norm:
                self.ln.append(nn.LayerNorm(layers[i + 1]))

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
                elif self.layer_norm:
                    x = self.ln[i](x)
            x = self.dropout(x)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(net = {self.net})"

class DotProdMAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, layer_norm=False, spectral_norm=False):
        super(DotProdMAB, self).__init__()
        self.dim_V = dim_V
        self.dim_Q = dim_Q
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if layer_norm:
            self.ln0 = nn.LayerNorm(dim_V)
        self.fc_o1 = nn.Linear(dim_V, dim_V)

        if spectral_norm:
            self.fc_q = SpectralNorm(self.fc_q)
            self.fc_k = SpectralNorm(self.fc_k)
            self.fc_v = SpectralNorm(self.fc_v)
            self.fc_o1 = SpectralNorm(self.fc_o1)

    def forward(self, Q, K, V, attn_mask=None, need_weights=False):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(V)

        head_dim = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(head_dim, 2), 0)
        K_ = torch.cat(K.split(head_dim, 2), 0)
        V_ = torch.cat(V.split(head_dim, 2), 0)
        logits = Q_.bmm(K_.transpose(1,2))/math.sqrt(head_dim)

        if attn_mask is not None:
            inf = torch.tensor(1e38, dtype=torch.float32, device=Q.device)
            logits = logits + (attn_mask) * -inf
        
        A = torch.softmax(logits, 2)
        O = torch.cat((A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = self.fc_o1(O)

        if need_weights:
            return [O, A]
        return [O]

# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class MAB(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_output_dim: int,
        ff_layers: list = [],
        use_custom_mab: bool = False,
        conditioning: bool = False,
        layer_norm: bool = False,
        spectral_norm: bool = False,
        dropout_p: float = 0.0,
        final_linear: bool = True,
        linear_args={},
    ):
        super(MAB, self).__init__()

        self.num_heads = num_heads
        if use_custom_mab:
            self.attention = DotProdMAB(embed_dim, embed_dim, embed_dim, num_heads, layer_norm, spectral_norm)
        else:
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.conditioning = conditioning

        # Single linear layer to project from dim(x+z') to dim(x)
        if self.conditioning:
            self.attn_ff   = nn.Linear(embed_dim, ff_output_dim)

        self.ff = LinearNet(
            ff_layers,
            input_size=ff_output_dim,
            output_size=ff_output_dim,
            final_linear=final_linear,
            **linear_args,
        )

        self.layer_norm = layer_norm
        self.spectral_norm = spectral_norm

        if self.layer_norm:
            self.norm1 = nn.LayerNorm(ff_output_dim)
            self.norm2 = nn.LayerNorm(ff_output_dim)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor, y: Tensor, y_mask: Tensor = None, z: Tensor = None):
        if y_mask is not None:
            # torch.nn.MultiheadAttention needs a mask of shape [batch_size * num_heads, N, N]
            y_mask = torch.repeat_interleave(y_mask, self.num_heads, dim=0)

        # Concatenate q,k,v inputs with conditioning vector if self.conditioning==True 
        # Linearly project output (dim(x+z')) back to dim(x)
        if self.conditioning:
            assert z is not None
            # Concat z with query
            x_ = torch.cat((x, z.unsqueeze(1).repeat(1, x.shape[1], 1)), dim=2)
            # Concat z with key/value
            y_ = torch.cat((y, z.unsqueeze(1).repeat(1, y.shape[1], 1)), dim=2)
            x = x + self.attn_ff(self.attention(x_, y_, y_, attn_mask=y_mask, need_weights=False)[0])
        else:
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

    def forward(self, x: Tensor, mask: Tensor = None, z: Tensor = None):
        if mask is not None:
            # torch.nn.MultiheadAttention needs a mask vector for each target node
            # i.e. reshaping from [B, N, 1] -> [B, N, N]
            mask = mask.transpose(-2, -1).repeat((1, mask.shape[-2], 1))

        return self.mab(x, x, mask, z)


# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class PMA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_seeds: int,
        **mab_args,
    ):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, mab_args['ff_output_dim']))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(embed_dim, **mab_args)

    def forward(self, x: Tensor, mask: Tensor = None, z: Tensor = None):
        if mask is not None:
            mask = mask.transpose(-2, -1)

        return self.mab(self.S.repeat(x.size(0), 1, 1), x, mask, z)


# Adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
class ISAB(nn.Module):
    def __init__(self, num_inds, embed_dim, **mab_args):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, mab_args['ff_output_dim']))
        self.num_inds = num_inds
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(embed_dim=embed_dim, **mab_args)
        self.mab1 = MAB(embed_dim=embed_dim, **mab_args)

    def forward(self, X, mask: Tensor = None, z: Tensor = None):
        if mask is not None:
            mask = mask.transpose(-2, -1).repeat((1, self.num_inds, 1))
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask, z)
        
        return self.mab1(X, H, z=z)

class ISE(nn.Module):
    def __init__(
        self,
        num_inds: int,
        embed_dim: int,
        **mab_args
    ):
        super(ISE, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, mab_args['ff_output_dim']))
        self.num_inds = num_inds
        nn.init.xavier_uniform_(self.I)
        self.mab = MAB(embed_dim, **mab_args)
    
    def forward(self, x: Tensor, mask: Tensor = None, z: Tensor = None):
        if mask is not None:
            mask = mask.transpose(-2, -1).repeat((1, self.num_inds, 1))
        H = self.mab(self.I.repeat(x.size(0), 1, 1), x, mask, z)

        return H.sum(dim=1)


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
        global_noise_input_dim: int = 0,
        global_noise_feat_dim: int = 0,
        global_noise_layers: list = [],
        learnable_init_noise: bool = False,
        noise_conditioning: bool = False,
        n_conditioning: bool = False,
        n_normalized: bool = False,
        sab_layers: int = 2,
        use_custom_mab: bool = False,
        num_heads: int = 4,
        embed_dim: int = 32,
        init_noise_dim: int = 8,
        sab_fc_layers: list = [],
        layer_norm: bool = False,
        spectral_norm: bool = False,
        dropout_p: float = 0.0,
        final_fc_layers: list = [],
        use_mask: bool = True,
        use_isab: bool = False,
        num_isab_nodes: int = 10,
        block_residual = False,
        linear_args: dict = {},
    ):
        super(GAPT_G, self).__init__()
        self.num_particles = num_particles
        self.output_feat_size = output_feat_size
        self.use_mask = use_mask
        self.learnable_init_noise = learnable_init_noise
        self.noise_conditioning = noise_conditioning
        self.n_conditioning = n_conditioning
        self.n_normalized = n_normalized
        self.block_residual = block_residual
        # Learnable gaussian noise for sampling initial set
        if self.learnable_init_noise:
            self.mu = nn.Parameter(torch.randn(self.num_particles, init_noise_dim))
            self.std = nn.Parameter(torch.randn(self.num_particles, init_noise_dim))

            # Projecting initial noise z to embed_dims
            # self.input_embedding = LinearNet(
            #     layers = [],
            #     input_size = init_noise_dim,
            #     output_size = embed_dim,
            #     **linear_args
            # )

        # MLP for processing conditioning vector (input dims = global noise dims + 1)
        if noise_conditioning or n_conditioning:
            noise_net_input_dim = 0
            if noise_conditioning:
                noise_net_input_dim += global_noise_input_dim
            if n_conditioning:
                noise_net_input_dim += 1
            self.global_noise_net = LinearNet(
                layers = global_noise_layers,
                input_size = noise_net_input_dim,
                output_size = global_noise_feat_dim,
                **linear_args
            )

        self.sabs = nn.ModuleList()

        # Adjust MAB input dims based on conditioning
        ff_output_dim = embed_dim
        if noise_conditioning or n_conditioning:
            embed_dim += global_noise_feat_dim

        sab_args = {
            "embed_dim": embed_dim,
            "ff_output_dim": ff_output_dim,
            "ff_layers": sab_fc_layers,
            "conditioning": noise_conditioning or n_conditioning,
            "final_linear": False,
            "num_heads": num_heads,
            "use_custom_mab": use_custom_mab,
            "layer_norm": layer_norm,
            "spectral_norm": spectral_norm,
            "dropout_p": dropout_p,
            "linear_args": linear_args,
        }

        # intermediate layers
        for _ in range(sab_layers):
            self.sabs.append(SAB(**sab_args) if not use_isab else ISAB(num_isab_nodes, **sab_args))

        self.final_fc = LinearNet(
            final_fc_layers,
            input_size=ff_output_dim,
            output_size=output_feat_size,
            final_linear=True,
            **linear_args,
        )

    def forward(self, x: Tensor, labels: Tensor = None, z: Tensor = None):
        if self.use_mask:
            # unnormalize the last jet label - the normalized # of particles per jet
            # (between 1/``num_particles`` and 1) - to between 0 and ``num_particles`` - 1
            num_jet_particles = (labels[:, -1] * self.num_particles).int() - 1
            # sort the particles by the first noise feature per particle, and the first
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
        
        # if self.learnable_init_noise:
        #     x = self.input_embedding(x)
        
        # Concatenate global noise and # particles depending on conditioning
        if self.n_normalized:
            num_jet_particles = labels[:, -1]
        else:
            num_jet_particles += 1
        if self.noise_conditioning or self.n_conditioning:
            if self.noise_conditioning and self.n_conditioning:
                z = torch.cat((z, num_jet_particles.unsqueeze(1)), dim=1)
            elif self.n_conditioning:
                z = num_jet_particles.unsqueeze(1).float()
            z = self.global_noise_net(z)
        
        for sab in self.sabs:
            sab_out = sab(x, _attn_mask(mask), z)
            x = x + sab_out if self.block_residual else sab_out

        x = torch.tanh(self.final_fc(x))

        return torch.cat((x, mask - 0.5), dim=2) if mask is not None else x

    def sample_init_set(self, batch_size):
        if self.learnable_init_noise:
            cov = torch.eye(self.std.shape[1]).repeat(self.num_particles, 1, 1).to(self.std.device) * (self.std ** 2).unsqueeze(2)
            assert cov.shape==(self.num_particles, self.std.shape[1], self.std.shape[1])
            mvn = MultivariateNormal(loc=self.mu, covariance_matrix=cov)
            return mvn.rsample((batch_size, ))
            # std_mu = torch.zeros_like(self.mu).repeat(batch_size,1,1)
            # std_sigma = torch.ones_like(self.std).repeat(batch_size,1,1)
            # std_samples = torch.normal(std_mu, std_sigma)
            # return std_samples * self.std + self.mu


class GAPT_D(nn.Module):
    def __init__(
        self,
        num_particles: int,
        input_feat_size: int,
        sab_layers: int = 2,
        num_heads: int = 4,
        embed_dim: int = 32,
        cond_feat_dim: int = 8,
        cond_net_layers: list = [],
        n_conditioning: bool = False,
        n_normalized: bool = False,
        use_custom_mab: bool = False,
        sab_fc_layers: list = [],
        layer_norm: bool = False,
        spectral_norm: bool = False,
        dropout_p: float = 0.0,
        final_fc_layers: list = [],
        use_mask: bool = True,
        use_isab: bool = False,
        num_isab_nodes: int = 10,
        use_ise: bool = False,
        num_ise_nodes: int = 10,
        block_residual = False,
        linear_args: dict = {},
    ):
        super(GAPT_D, self).__init__()
        self.num_particles = num_particles
        self.input_feat_size = input_feat_size
        self.use_mask = use_mask
        self.n_conditioning = n_conditioning
        self.n_normalized = n_normalized
        self.use_ise = use_ise
        self.block_residual = block_residual
        # MLP for processing # particles
        if n_conditioning:
            cond_net_input_dim = 1
            self.cond_net = LinearNet(
                layers = cond_net_layers,
                input_size = cond_net_input_dim,
                output_size = cond_feat_dim,
                **linear_args
            )
        self.sabs = nn.ModuleList()

        ff_output_dim = embed_dim
        if n_conditioning:
            embed_dim += cond_feat_dim

        sab_args = {
            "embed_dim": embed_dim,
            "ff_layers": sab_fc_layers,
            "ff_output_dim": ff_output_dim,
            "use_custom_mab": use_custom_mab,
            "conditioning": n_conditioning,
            "final_linear": False,
            "num_heads": num_heads,
            "layer_norm": layer_norm,
            "spectral_norm": spectral_norm,
            "dropout_p": dropout_p,
            "linear_args": linear_args,
        }

        self.input_embedding = LinearNet(
            [], input_size=input_feat_size, output_size=ff_output_dim, **linear_args
        )

        # Intermediate layers
        for _ in range(sab_layers):
            self.sabs.append(SAB(**sab_args) if not use_isab else ISAB(num_isab_nodes, **sab_args))
        
        # Encoding/Pooling layers
        linear_net_input_dim = ff_output_dim
        if use_ise:
            self.ises = nn.ModuleList()
            for _ in range(sab_layers):
                self.ises.append(ISE(num_ise_nodes, **sab_args))
            linear_net_input_dim *= sab_layers
        else:
            self.pma = PMA(
                num_seeds=1,
                **sab_args,
            )

        # Classification network
        self.final_fc = LinearNet(
            final_fc_layers,
            input_size=linear_net_input_dim,
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
        
        # Use # particles for conditioning
        z = None
        if self.n_conditioning:
            if self.n_normalized:
                num_jet_particles = labels[:, -1]
            else:
                num_jet_particles = (labels[:, -1] * self.num_particles).int()
            z = num_jet_particles.unsqueeze(1).float()
            z = self.cond_net(z)
        
        # Use appropriate forward pass corresponding to encoding/pooling operation
        if self.use_ise:
            e = torch.Tensor().to(x.device)
            for sab, ise in zip(self.sabs, self.ises):
                sab_out = sab(x, _attn_mask(mask), z)
                x = x + sab_out if self.block_residual else sab_out
                e = torch.cat((e, ise(x, _attn_mask(mask), z)), dim=1)
            out = e
        else:
            for sab in self.sabs:
                sab_out = sab(x, _attn_mask(mask), z)
                x = x + sab_out if self.block_residual else sab_out
            out = self.pma(x, _attn_mask(mask), z).squeeze()
        
        return torch.sigmoid(self.final_fc(out))
