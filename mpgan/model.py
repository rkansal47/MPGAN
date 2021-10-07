import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spectral_normalization import SpectralNorm
import logging


class LinearNet(nn.Module):
    """
    Module for fully connected networks with leaky relu activations

    Args:
        layers (list): list with layers of the fully connected network, optionally containing the input and output sizes inside e.g. [input_size, ... hidden layers ..., output_size]
        input_size (list): size of input, if 0 or unspecified, first element of `layers` will be treated as the input size
        output_size (list): size of output, if 0 or unspecified, last element of `layers` will be treated as the output size
        final_linear (bool): keep the final layer operation linear i.e. no normalization, no nonlinear activation. Defaults to False.
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


class MPLayer(nn.Module):
    """
    MPLayer as described in Kansal et. al. *Particle Cloud Generation with Message Passing Generative Adversarial Networks* (https://arxiv.org/abs/2106.11535).
    TODO: mathematical formulation

    Args:
        node_size (int): input node feature size
        fe_layers (list): list of edge network intermediate and output layer sizes
        fn_layers (list): list of node network intermediate and output layer output sizes
        pos_diffs (bool): use some measure of the distance between nodes as the edge features between them. Defaults to False.
        all_ef (bool): use the euclidean distance between all the node features as an edge feature - only active is ``pos_diffs`` is True. Defaults to True.
        coords (str): the coordinate system used for node features ('polarrel', 'polar', or 'cartesian'), only active if ``delta_coords`` or ``delta_r`` is True. Defaults to "polarrel".
        delta_coords (bool): use the vector difference between the two nodes as edge features. Defaults to False.
        delta_r (bool): use the delta R between two nodes as edge features. Defaults to True.
        int_diffs (bool): **Not implemented yet!** use the difference between pT as an edge feature. Defaults to False.
        clabels (int): number of conditioning labels to use. Defaults to 0.
        mask_fne_np (bool): use number of particles per jet as conditional label. Defaults to False.
        fully_connected (bool): use fully connected graph for message passing. Defaults to True.
        num_knn (int): if not fully connected, number of nodes to use for knn for message passing. Defaults to 20.
        self_loops (bool): if not fully connected, allow for self loops in message passing. Defaults to True.
        sum (bool): sum as the message aggregation operation, as opposed to mean. Defaults to True.
        **linear_args: additional arguments for linear layers, given to LinearNet modules.

    """

    def __init__(
        self,
        node_size: int,
        fe_layers: list,
        fn_layers: list,
        pos_diffs: bool = False,
        all_ef: bool = True,
        coords: str = "polarrel",
        delta_coords: bool = False,
        delta_r: bool = True,
        int_diffs: bool = False,
        clabels: int = 0,
        mask_fne_np: bool = False,
        fully_connected: bool = True,
        num_knn: int = 20,
        self_loops: bool = True,
        sum: bool = True,
        **linear_args,
    ):
        super(MPLayer, self).__init__()

        self.node_size = node_size
        self.fe_layers = fe_layers
        self.fn_layers = fn_layers

        self.pos_diffs = pos_diffs
        self.all_ef = all_ef
        self.coords = coords
        self.delta_coords = delta_coords
        self.delta_r = delta_r
        self.int_diffs = int_diffs

        self.clabels = clabels
        self.mask_fne_np = mask_fne_np

        self.fully_connected = fully_connected
        self.num_knn = num_knn
        self.self_loops = self_loops
        self.sum = sum

        # number of edge features to pass into edge network (e.g. node distances, pT difference etc.)
        num_ef = 0
        if pos_diffs:
            if delta_coords:
                num_ef += 3 if coords == "cartesian" else 2
            if delta_r or all_ef:
                num_ef += 1  # currently can't add delta_r and all_ef edge features both together

        num_ef += int(int_diffs)
        self.num_ef = num_ef

        # edge network input is node 1 features + node 2 features + edge features (optional) + conditional labels (optional) + # particles (optional)
        fe_in_size = 2 * node_size + num_ef + clabels + mask_fne_np
        self.fe = LinearNet(self.fe_layers, input_size=fe_in_size, final_linear=False, **linear_args)

        # node network input is edge network output + node features + conditional labels (optional) + # particles (optional)
        fe_out_size = self.fe_layers[-1]
        fn_in_size = fe_out_size + node_size + clabels + mask_fne_np
        # node network output is 'linear' i.e. final layer does not apply normalization or nonlinear activations
        self.fn = LinearNet(self.fn_layers, input_size=fn_in_size, final_linear=True, **linear_args)

    def forward(self, x: Tensor, use_mask: bool = False, mask: Tensor = None, labels: Tensor = None, num_jet_particles: Tensor = None):
        """
        Runs through message passing. Has optional arguments for masking and conditioning.

        Args:
            x (Tensor): input tensor of shape ``[batch size, # nodes, # node features]``
            use_mask (bool, optional): use mask to ignore zero-masked particles during message passing.
            mask (Tensor, optional): if using masking, tensor of masks for each node of shape ``[batch size, # nodes, 1 (mask)]``
            labels (Tensor, optional): if using conditioning labels during message passing, tensor of labels for each jet of shape [batch size, # labels]
            num_jet_particles (Tensor, optional): if using # of particles as an extra conditioning label, tensor of num particles for each jet of shape [batch size, 1]
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)

        assert not (use_mask and mask is None), "need ``mask`` tensor if using ``use_mask`` option"
        assert not (self.clabels and labels is None), "need ``labels`` tensor if using ``clabels`` option"
        assert not (self.mask_fne_np and num_jet_particles is None), "need ``num_jet_particles`` tensor if using ``mask_fne_np`` option"

        # get inputs to edge network
        if self.fully_connected:
            A, A_mask = self._getA_fully_connected(x, batch_size, num_nodes, use_mask, mask)
            num_knn = num_nodes  # if fully connected num_knn is the size of the graph
        else:
            A, A_mask = self._getA_knn(x, batch_size, num_nodes, use_mask, mask)
            num_knn = self.num_knn

        if self.clabels:
            # add conditioning labels
            A = torch.cat((A, labels[:, : self.clabels].repeat(num_nodes * num_knn, 1)), axis=1)

        if self.mask_fne_np:
            # add # of real (i.e. not zero-padded) particles in the graph
            A = torch.cat((A, num_jet_particles.repeat(num_nodes * num_knn, 1)), axis=1)

        # run through edge network
        A = self.fe(A)
        A = A.view(batch_size, num_nodes, num_knn, self.fe_layers[-1])

        if use_mask:
            # if use masking, mask out 0-masked particles by multiplying them with the mask
            if self.fully_connected:
                A = A * mask.unsqueeze(1)
            else:
                A = A * A_mask.view(batch_size, self.args.num_hits, num_knn, 1)

        # aggregate and concatenate with node features
        A = torch.sum(A, 2) if self.sum else torch.mean(A, 2)
        x = torch.cat((A, x), 2).view(batch_size * num_nodes, -1)

        if self.clabels:
            # add conditioning labels
            x = torch.cat((x, labels[:, : self.clabels].repeat(num_nodes, 1)), axis=1)

        if self.mask_fne_np:
            # add # of real (i.e. not zero-padded) particles in the graph
            x = torch.cat((x, num_jet_particles.repeat(num_nodes, 1)), axis=1)

        # run through node network
        x = self.fn(x)
        x = x.view(batch_size, num_nodes, self.fn_layers[-1])

        return x

    def _getA_fully_connected(self, x, batch_size, num_nodes, use_mask, mask):
        """
        returns tensor of inputs to the edge networks using a fully connected graph
        """
        num_coords = 3 if self.coords == "cartesian" else 2
        out_size = 2 * self.node_size + self.num_ef
        node_size = x.shape[2]

        A_mask = None

        x1 = x.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, node_size)
        x2 = x.repeat(1, num_nodes, 1)

        if self.pos_diffs:
            # get the extra edge features for the edge networks
            if self.all_ef:
                diffs = x2 - x1
            else:
                diffs = x2[:, :, :num_coords] - x1[:, :, :num_coords]

            dists = torch.norm(diffs + 1e-12, dim=2).unsqueeze(2)

            if self.delta_r and self.delta_coords:
                A = torch.cat((x1, x2, diffs, dists), 2)
            elif self.delta_r or self.all_ef:
                A = torch.cat((x1, x2, dists), 2)
            elif self.delta_coords:
                A = torch.cat((x1, x2, diffs), 2)

            A = A.view(batch_size * num_nodes * num_nodes, out_size)
        else:
            A = torch.cat((x1, x2), 2).view(batch_size * num_nodes * num_nodes, out_size)

        return A, A_mask

    def _getA_knn(self, x, batch_size, num_nodes, use_mask, mask):
        """
        returns tensor of inputs to the edge networks by finding the k-nearest-neighbours for each node
        """
        num_coords = 3 if self.coords == "cartesian" else 2
        out_size = 2 * self.node_size + self.num_ef
        node_size = x.shape[2]

        A_mask = None

        x1 = x.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, node_size)

        if use_mask:
            # multiply masked particles by this so they are not selected as a nearest neighbour
            mul = 1e4
            x2 = (((1 - mul) * mask + mul) * x).repeat(1, num_nodes, 1)
        else:
            x2 = x.repeat(1, num_nodes, 1)

        # get dists between each pair of nodes
        if self.all_ef or not self.pos_diffs:
            diffs = x2 - x1
        else:
            diffs = x2[:, :, :num_coords] - x1[:, :, :num_coords]

        dists = torch.norm(diffs + 1e-12, dim=2).reshape(batch_size, num_nodes, num_nodes)

        # sort the distances to find the k-nearest neighbours
        sorted = torch.sort(dists, dim=2)
        # if self_loops is True then 0 else 1 so that we skip the node itself in the line below if no self loops
        self_loops_idx = int(self.self_loops is False)

        # ``dists`` contains the sorted distances between pair of nodes, ``sorted`` the indices of the nodes
        dists = sorted[0][:, :, self_loops_idx : self.num_knn + self_loops_idx].reshape(batch_size, num_nodes * self.num_knn, 1)
        sorted = sorted[1][:, :, self_loops_idx : self.num_knn + self_loops_idx].reshape(batch_size, num_nodes * self.num_knn, 1)
        sorted.reshape(batch_size, num_nodes * self.num_knn, 1).repeat(1, 1, node_size)

        x1_knn = x.repeat(1, 1, self.num_knn).view(batch_size, num_nodes * self.num_knn, node_size)

        # gather the k nearest neighbours using the ``sorted`` tensor containing their indices
        if use_mask:
            x2_knn = torch.gather(torch.cat((x, mask), dim=2), 1, sorted.repeat(1, 1, node_size + 1))
            A_mask = x2_knn[:, :, -1:]
            x2_knn = x2_knn[:, :, :-1]
        else:
            x2_knn = torch.gather(x, 1, sorted.repeat(1, 1, node_size))

        # finally get A tensor containing each node and its nearest neighbour + optionally the distance between them
        if self.pos_diffs:
            A = torch.cat((x1_knn, x2_knn, dists), dim=2)
        else:
            A = torch.cat((x1_knn, x2_knn), dim=2)

        return A, A_mask

    def __repr__(self):
        return f"{self.__class__.__name__}(fe = {self.fe}, \n fn = {self.fn})"


class MPNet(nn.Module):
    """
    Generic base class for a message passing network, inherited by MPGenerator and MPDiscriminator networks.

    Performs ``mp_iters`` iterations of message passing using the ``MPLayer`` module.
    Arguments for the ``MPLayer`` and ``LinearNet`` modules are inputed separately via the ``mp_args`` and ``linear_args`` dict.

    Args:
        num_particles (int): max number of particles per jet.
        input_node_size (int): number of input features per particle.
        mp_iters (int): number of message passing iterations. Defaults to 2.
        fe_layers (list): ``MPLayer``s edge network layer sizes. Defaults to [96, 160, 192].
        fn_layers (list): ``MPLayer``s node network layer sizes. Defaults to [256, 256].
        fe1_layers (list): edge network layer sizes for the first MPLayer, if different from the rest (i.e. ``fe_layers``).
        fn1_layers (list): node network layer sizes for the first MPLayer, if different from the rest (``fm_layers``).
        hidden_node_size (int): intermediate number of node features during message passing. Defaults to 32.
        output_node_size (int): number of desired output features per particle. If not specified, same as ``hidden_node_size``.
        final_activation (str): final activation function to use. Options are 'sigmoid', 'tanh' or nothing (''). Defaults to "".
        linear_args (dict): dict of args for ``LinearNet`` module.
        mp_args (dict): dict of args for ``MPLayer`` module.
        mp_args_first_layer (dict): dict of args for the first ``MPLayer`` layer, if different from the rest.
        mask_args (dict): dict of mask-related args. Defined in the mask functions for the individual networks below.
    """

    def __init__(
        self,
        num_particles: int,
        input_node_size: int,
        mp_iters: int = 2,
        fe_layers: list = [96, 160, 192],
        fn_layers: list = [256, 256],
        fe1_layers: list = None,
        fn1_layers: list = None,
        hidden_node_size: int = 32,
        output_node_size: int = 0,
        final_activation: str = "",
        linear_args: dict = {},
        mp_args: dict = {},
        mp_args_first_layer: dict = {},
        mask_args: dict = {},
    ):
        super(MPNet, self).__init__()
        self.num_particles = num_particles
        self.input_node_size = input_node_size
        self.output_node_size = output_node_size if output_node_size > 0 else hidden_node_size
        self.mp_iters = mp_iters

        fe1_layers = fe_layers if fe1_layers is None else fe1_layers
        fn1_layers = fn_layers if fn1_layers is None else fn1_layers

        self.hidden_node_size = hidden_node_size
        self.final_activation = final_activation

        self.linear_args = linear_args

        # copy all keys not specified in ``mp_args_first_layer`` dict from ``mp_args` dict
        for key in mp_args:
            if key not in mp_args_first_layer:
                mp_args_first_layer[key] = mp_args[key]

        self.mask_args = mask_args

        self._init_mask(**mask_args)

        self.mp_layers = nn.ModuleList()

        self.mp_layers.append(MPLayer(input_node_size, fe1_layers, fn1_layers, **mp_args_first_layer, **linear_args))

        # intermediate layers
        for i in range(mp_iters - 2):
            self.mp_layers.append(MPLayer(hidden_node_size, fe_layers, fn_layers, **mp_args, **linear_args))

        # final layer; specifying final node size TODO: only make this one final_linear
        fn_layers[-1] = output_node_size
        self.mp_layers.append(MPLayer(hidden_node_size, fe_layers, fn_layers, **mp_args, **linear_args))

    def forward(self, x: Tensor, labels: Tensor = None) -> Tensor:
        """Forward pass of MPNet including optional pre and post processing and optional masking.

        Args:
            x (Tensor): input data tensor of shape ``[batch_size, ...]`` where size depends on the particular implementation.
            labels (Tensor): optional ensor of jet level features for a conditioning and/or masking of shape ``[batch_size, num_jet_features]``.

        Returns:
            Tensor: transformed tensor.

        """
        x = self._pre_mp(x, labels)

        x, use_mask, mask, num_jet_particles = self._get_mask(x, **self.mask_args)

        # message passing
        for i in range(self.mp_iters):
            x = self.mp_layers[i](x, use_mask, mask, labels, num_jet_particles)

        x = self._post_mp(x, labels, use_mask, mask, num_jet_particles)
        x = self._final_activation(x)
        x = self._final_mask(x, mask, **self.mask_args)

        return x

    def _pre_mp(self, x, labels):
        """Pre-message-passing operations"""
        return x

    def _post_mp(self, x, labels, use_mask, mask, num_jet_particles):
        """Post-message-passing operations"""
        return x

    def _final_activation(self, x):
        """Apply the final activation to the network's output"""
        if self.final_activation == "tanh":
            x = torch.tanh(x)
        elif self.final_activation == "sigmoid":
            x = torch.sigmoid(x)

        return x

    def _init_mask(self, **mask_args):
        """
        Intialize potential mask networks and variables.
        """
        return

    def _get_mask(self, x: Tensor, **mask_args):
        """
        Develops mask for input tensor ``x`` depending on the chosen masking strategy.
        """
        return x, False, None, None

    def _final_mask(self, x: Tensor, **mask_args):
        """
        Perform any final mask operations.
        """
        return x

    def __repr__(self):
        return f"MPLayers = {self.mp_layers})"


class MPGenerator(MPNet):
    """
    Message passing generator.
    Goes through an optional latent fully connected layer then ``mp_iters`` iterations of message passing to output a tensor of shape ``[batch_size, num_particles, output_node_size]``.

    A number of options for masking are implemented, as described in the appendix of
    Kansal et. al. *Particle Cloud Generation with Message Passing Generative Adversarial Networks* (https://arxiv.org/abs/2106.11535).
    Args for masking are described in the masking functions below.

    Input ``x`` tensor to the forward pass must be of shape ``[batch_size, lfc_latent_size]`` if using ``lfc`` else ``[batch_size, num_particles, input_node_size]``.

    Args:
        lfc (bool): use a fully connected network to go from a vector latent space to a graph structure of ``num_particles`` nodes with ``node_input_size`` features. Defaults to False.
        lfc_latent_size (int): if using ``lfc``, size of the vector latent space. Defaults to 128.
        **mpnet_args: args for ``MPNet`` base class.
    """

    def __init__(self, lfc: bool = False, lfc_latent_size: int = 128, **mpnet_args):
        super(MPGenerator, self).__init__(**mpnet_args)

        # latent fully connected layer
        self.lfc = lfc
        if lfc:
            self.lfc_layer = nn.Linear(lfc_latent_size, self.num_particles * self.input_node_size)

    def _pre_mp(self, x, labels):
        """Pre-message-passing operations"""
        if self.lfc:
            x = self.lfc_layer(x).reshape(x.shape[0], self.num_particles, self.input_layer_node_size)

    def _init_mask(self, mask_learn: bool = False, mask_learn_sep: bool = False, fmg: list = [64], **mask_args):
        """
        Intialize potential mask networks and variables.

        Args:
            mask_learn (bool): learning a mask per particle using each particle's initial noise. Defaults to False.
            mask_learn_sep (bool): predicting an overall number of particles per jet using separate jet noise. Defaults to False.
            fmg (list): list of mask network intermediate layer sizes. Defaults to [64].
            **mask_args: extra mask args not needed for this function.
        """

        if mask_learn or mask_learn_sep:
            self.fmg_layer = LinearNet(
                fmg,
                input_size=self.first_layer_node_size,
                output_size=1 if mask_learn else self.num_particles,
                final_linear=True,
                **self.linear_args,
            )

    def _get_mask(
        self,
        x: Tensor,
        labels: Tensor = None,
        mask_learn: bool = False,
        mask_learn_bin: bool = True,
        mask_learn_sep: bool = False,
        mask_c: bool = True,
        mask_fne_np: bool = False,
        **mask_args,
    ):
        """
        Develops mask for input tensor ``x`` depending on the chosen masking strategy.

        Args:
            x (Tensor): input tensor.
            labels (Tensor): input jet level features - last feature should be # of particles in jet if ``mask_c``. Defaults to None.
            mask_learn (bool): learning a mask per particle using each particle's initial noise. Defaults to False.
            mask_learn_bin (bool): learn a binary mask as opposed to continuous. Defaults to True.
            mask_learn_sep (bool): predicting an overall number of particles per jet using separate jet noise. Defaults to False.
            mask_c (bool): using input # of particles per jet to automatically choose masks for particles. Defaults to True.
            mask_fne_np (bool): feed # of particle per jet as an input to the node and edge networks. Defaults to False.
            **mask_args: extra mask args not needed for this function.

        Returns:
            x (Tensor): modified input tensor
            use_mask (bool): is masking being used
            mask (Tensor): if ``use_mask`` then tensor of masks of shape ``[batch size, # nodes, 1 (mask)]``, else None
            num_jet_particles (Tensor): if ``use_mask`` then tensor of # of particles per jet of shape ``[batch size, 1 (num particles)]``, else None

        """

        use_mask = mask_learn or mask_c or mask_learn_sep

        if not use_mask:
            return x, use_mask, None, None

        num_jet_particles = None

        if mask_learn:
            # predict a mask from the noise per particle using the fmg fully connected network
            mask = self.fmg_layer(x)
            # sign function if learning a binary mask else sigmoid
            mask = torch.sign(mask) if mask_learn_bin else torch.sigmoid(mask)

            if mask_fne_np:
                # num_jet_particles will be an extra feature inputted to the edge and node networks
                num_jet_particles = torch.mean(mask, dim=1)
                logging.debug("num_jet_particles \n {}".format(num_jet_particles[:2]))

        elif mask_c:
            # unnormalize the last jet label - the normalized # of particles per jet (between 1/``num_particles`` and 1) - to between 0 and ``num_particles`` - 1
            num_jet_particles = (labels[:, -1] * self.num_particles).int() - 1
            # sort the particles bythe first noise feature per particle, and the first ``num_jet_particles`` particles receive a 1-mask, the rest 0.
            mask = (x[:, :, 0].argsort(1).argsort(1) <= num_jet_particles.unsqueeze(1)).unsqueeze(2).float()
            logging.debug("x \n {} \n num particles \n {} \n gen mask \n {}".format(x[:2, :, 0], num_jet_particles[:2], mask[:2, :, 0]))

        elif mask_learn_sep:
            # last 'particle' in tensor is input to the fmg ``num_jet_particles`` prediction network
            num_jet_particles_input = x[:, -1, :]
            x = x[:, :-1, :]

            num_jet_particles = self.fmg_layer(num_jet_particles_input)
            num_jet_particles = torch.argmax(num_jet_particles, dim=1)
            # sort the particles by the first noise feature per particle, and the first ``num_jet_particles`` particles receive a 1-mask, the rest 0.
            mask = (x[:, :, 0].argsort(1).argsort(1) <= num_jet_particles.unsqueeze(1)).unsqueeze(2).float()

        return x, use_mask, mask, num_jet_particles

    def _final_mask(
        self,
        x: Tensor,
        mask: Tensor,
        mask_feat_bin: bool = False,
        **mask_args,
    ):
        """
        Process the output to get the final mask.

        Args:
            x (Tensor): processed data tensor.
            mask (Tensor): mask tensor, if being used in this model.
            mask_feat_bin (bool): use the last output feature as a binary mask. Defaults to False.
            **mask_args: extra mask args not needed for this function.

        Returns:
            type: final ``x`` tensor possibly including the mask as the last feature.

        """

        if mask_feat_bin:
            # take last output feature and make it binary
            mask = x[:, :, -1]
            x = x[:, :, :-1]

            if mask_feat_bin:
                mask = torch.sign(mask)

        return torch.cat((x, mask - 0.5), dim=2) if mask is not None else x

    def __repr__(self):
        lfc_str = f"LFC = {self.lfc_layer},\n" if self.lfc else ""
        fmg_str = f"FMG = {self.fmg_layer},\n" if hasattr(self, "fmg_layer") else ""
        return f"{self.__class__.__name__}({lfc_str}{fmg_str}MPLayers = {self.mp_layers})"


class MPDiscriminator(MPNet):
    """
    Message passing discriminator.
    Goes through ``mp_iters`` iterations of message passing and then an optional final fully connected network to output a scalar prediction.

    A number of options for masking are implemented, as described in the appendix of
    Kansal et. al. *Particle Cloud Generation with Message Passing Generative Adversarial Networks* (https://arxiv.org/abs/2106.11535).
    Args for masking are described in the masking functions below.

    Input ``x`` tensor to the forward pass must be of shape ``[batch_size, num_particles, input_node_size]``.

    Args:
        dea (bool): 'discriminator early aggregation' i.e. aggregate the final graph and pass through a final fully connected network ``fnd``. Defaults to True.
        dea_sum (bool): if using ``dea``, use 'sum' as the aggregation operation as opposed to 'mean'. Defaults to True.
        fnd (list): list of final FC network intermediate layer sizes. Defaults to [].
        mask_fnd_np (bool): pass number of particles as an extra feature into the final FC network. Defaults to False.
        **mpnet_args: args for ``MPNet`` base class.
    """

    def __init__(
        self,
        dea: bool = True,
        dea_sum: bool = True,
        fnd: list = [],
        mask_fnd_np: bool = False,
        **mpnet_args,
    ):
        super(MPDiscriminator, self).__init__(output_node_size=1 if not dea else 0, **mpnet_args)

        self.dea = dea
        self.dea_sum = dea_sum

        self.mask_fnd_np = mask_fnd_np

        # final fully connected classification layer
        if dea:
            self.fnd_layer = LinearNet(
                fnd, input_size=self.args.hidden_node_size + int(mask_fnd_np), output_size=1, final_linear=True, **self.linear_args
            )

    def _post_mp(self, x, labels, use_mask, mask, num_jet_particles):
        do_mean = not (self.dea and self.dea_sum)  # only summing if using ``dea`` and ``dea_sum`` is True
        if use_mask:
            # only sum contributions from 1-masked particles
            x = x * mask
            x = torch.sum(x, 1)
            if do_mean:
                # only divide by number of 1-masked particle per jet
                x = x / (torch.sum(mask, 1) + 1e-12)
        else:
            x = torch.mean(x, 1) if do_mean else torch.sum(x, 1)

        # feed into optional final FC network
        if self.dea:
            if self.mask_fnd_np:
                x = torch.cat((num_jet_particles, x), dim=1)

            x = self.fnd_layer(x)

        return x

    def _get_mask(
        self,
        x: Tensor,
        mask_manual: bool = False,
        mask_learn: bool = False,
        mask_learn_sep: bool = False,
        mask_c: bool = True,
        mask_fne_np: bool = False,
        mask_fnd_np: bool = False,
        **mask_args,
    ):
        """
        Develops mask for input tensor ``x`` depending on the chosen masking strategy.

        Args:
            x (Tensor): input tensor.
            mask_manual (bool): applying a manual mask after generation per particle based on a pT cutoff.
            mask_learn (bool): learning a mask per particle using each particle's initial noise. Defaults to False.
            mask_learn_sep (bool): predicting an overall number of particles per jet using separate jet noise. Defaults to False.
            mask_c (bool): using input # of particles per jet to automatically choose masks for particles. Defaults to True.
            mask_fne_np (bool): feed # of particle per jet as an input to the node and edge networks. Defaults to False.
            mask_fnd_np (bool): feed # of particle per jet as an input to final discriminator FC network. Defaults to False.
            **mask_args: extra mask args not needed for this function.

        Returns:
            x (Tensor): modified data tensor
            use_mask (bool): is masking being used
            mask (Tensor): if ``use_mask`` then tensor of masks of shape ``[batch size, # nodes, 1 (mask)]``, else None
            num_jet_particles (Tensor): if ``use_mask`` then tensor of # of particles per jet of shape ``[batch size, 1 (num particles)]``, else None

        """

        mask = None
        num_jet_particles = None

        use_mask = mask_manual or mask_learn or mask_c or mask_learn_sep

        # separate mask from other features
        if use_mask or mask_fnd_np:
            mask = x[:, :, -1:] + 0.5

        if use_mask:
            x = x[:, :, :-1]

        if mask_fne_np:
            num_jet_particles = torch.mean(mask, dim=1)
            logging.debug("num_jet_particles \n {}".format(num_jet_particles[:2]))

        return x, use_mask, mask, num_jet_particles

    def __repr__(self):
        dea_str = f",\nFND = {self.fnd_layer}" if self.fnd else ""
        return f"{self.__class__.__name__}(MPLayers = {self.mp_layers}{dea_str})"
