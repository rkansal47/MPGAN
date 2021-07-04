import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import NNConv
from torch_cluster import knn_graph

import logging

class rGANG(nn.Module):
    def __init__(self, args):
        super(rGANG, self).__init__()
        self.args = args

        self.args.rgang_fc.insert(0, self.args.latent_dim)

        layers = []
        for i in range(len(self.args.rgang_fc) - 1):
            layers.append(nn.Linear(self.args.rgang_fc[i], self.args.rgang_fc[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))

        layers.append(nn.Linear(self.args.rgang_fc[-1], self.args.num_hits * self.args.node_feat_size))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

        logging.info("rGAN generator: \n {}".format(self.model))

    def forward(self, x, labels=None, epoch=None):
        return self.model(x).reshape(-1, self.args.num_hits, self.args.node_feat_size)


class rGAND(nn.Module):
    def __init__(self, args):
        super(rGAND, self).__init__()
        self.args = args

        self.args.rgand_sfc.insert(0, self.args.node_feat_size)

        layers = []
        for i in range(len(self.args.rgand_sfc) - 1):
            layers.append(nn.Conv1d(self.args.rgand_sfc[i], self.args.rgand_sfc[i + 1], 1))
            layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))

        self.sfc = nn.Sequential(*layers)

        self.args.rgand_fc.insert(0, self.args.rgand_sfc[-1])

        layers = []
        for i in range(len(self.args.rgand_fc) - 1):
            layers.append(nn.Linear(self.args.rgand_fc[i], self.args.rgand_fc[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))

        layers.append(nn.Linear(self.args.rgand_fc[-1], 1))
        layers.append(nn.Sigmoid())

        self.fc = nn.Sequential(*layers)

        logging.info("rGAND sfc: \n {}".format(self.sfc))
        logging.info("rGAND fc: \n {}".format(self.fc))

    def forward(self, x, labels=None, epoch=None):
        x = x.reshape(-1, self.args.node_feat_size, 1)
        x = self.sfc(x)
        x = torch.max(x.reshape(-1, self.args.num_hits, self.args.rgand_sfc[-1]), 1)[0]
        return self.fc(x)


class GraphCNNGANG(nn.Module):
    def __init__(self, args):
        super(GraphCNNGANG, self).__init__()
        self.args = args

        self.dense = nn.Linear(self.args.latent_dim, self.args.num_hits * self.args.graphcnng_layers[0])

        self.layers = nn.ModuleList()
        self.edge_weights = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(self.args.graphcnng_layers) - 1):
            self.edge_weights.append(nn.Linear(self.args.graphcnng_layers[i], self.args.graphcnng_layers[i] * self.args.graphcnng_layers[i + 1]))
            self.layers.append(NNConv(self.args.graphcnng_layers[i], self.args.graphcnng_layers[i + 1], self.edge_weights[i], aggr='mean', root_weight=True, bias=True))
            self.bn_layers.append(torch_geometric.nn.BatchNorm(self.args.graphcnng_layers[i + 1]))

        self.edge_weights.append(nn.Linear(self.args.graphcnng_layers[-1], self.args.graphcnng_layers[-1] * self.args.node_feat_size))
        self.layers.append(NNConv(self.args.graphcnng_layers[-1], self.args.node_feat_size, self.edge_weights[-1], aggr='mean', root_weight=True, bias=True))
        self.bn_layers.append(torch_geometric.nn.BatchNorm(self.args.node_feat_size))

        logging.info("dense: ")
        logging.info(self.dense)

        logging.info("edge_weights: ")
        logging.info(self.edge_weights)

        logging.info("layers: ")
        logging.info(self.layers)

        logging.info("bn layers: ")
        logging.info(self.bn_layers)


    def forward(self, x, labels=None, epoch=None):
        x = F.leaky_relu(self.dense(x), negative_slope=self.args.leaky_relu_alpha)

        batch_size = x.size(0)
        x = x.reshape(batch_size * self.args.num_hits, self.args.graphcnng_layers[0])
        zeros = torch.zeros(batch_size * self.args.num_hits, dtype=int).to(self.args.device)
        zeros[torch.arange(batch_size) * self.args.num_hits] = 1
        batch = torch.cumsum(zeros, 0) - 1

        loop = self.args.num_knn == self.args.num_hits

        for i in range(len(self.layers)):
            edge_index = knn_graph(x, self.args.num_knn, batch, loop)
            edge_attr = x[edge_index[0]] - x[edge_index[1]]
            x = self.bn_layers[i](self.layers[i](x, edge_index, edge_attr))
            if i < (len(self.layers) - 1): x = F.leaky_relu(x, negative_slope=self.args.leaky_relu_alpha)

        if self.args.graphcnng_tanh: x = F.tanh(x)

        return x.reshape(batch_size, self.args.num_hits, self.args.node_feat_size)


class PointNetMixD(nn.Module):
    def __init__(self, args):
        super(PointNetMixD, self).__init__()
        self.args = args

        self.args.pointnetd_pointfc.insert(0, self.args.node_feat_size)

        self.args.pointnetd_fc.insert(0, self.args.pointnetd_pointfc[-1] * 2)
        self.args.pointnetd_fc.append(1)

        layers = []

        for i in range(len(self.args.pointnetd_pointfc) - 1):
            layers.append(nn.Linear(self.args.pointnetd_pointfc[i], self.args.pointnetd_pointfc[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))

        self.pointfc = nn.Sequential(*layers)

        layers = []

        for i in range(len(self.args.pointnetd_fc) - 1):
            layers.append(nn.Linear(self.args.pointnetd_fc[i], self.args.pointnetd_fc[i + 1]))
            if i < len(self.args.pointnetd_fc) - 2: layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))

        layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)


        logging.info("point fc: ")
        logging.info(self.pointfc)

        logging.info("fc: ")
        logging.info(self.fc)


    def forward(self, x, labels=None, epoch=None):
        batch_size = x.size(0)
        if self.args.mask:
            x[:, :, 2] += 0.5
            mask = x[:, :, 3:4] >= 0
            x = (x * mask)[:, :, :3]
            x[:, :, 2] -= 0.5
            print(mask[0])
            print(x[0])
        x = self.pointfc(x.view(batch_size * self.args.num_hits, self.args.node_feat_size)).view(batch_size, self.args.num_hits, self.args.pointnetd_pointfc[-1])
        x = torch.cat((torch.max(x, dim=1)[0], torch.mean(x, dim=1)), dim=1)
        return self.fc(x)




from torch.nn.modules.utils import _pair

# from https://discuss.pytorch.org/t/locally-connected-layers/26979
class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=True):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class LAGAND(nn.Module):
    def __init__(self, args):
        super(LAGAND, self).__init__()
        self.args = args

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),

        )


        self.dense = nn.Linear(self.args.latent_dim, self.args.num_hits * self.args.graphcnng_layers[0])

        self.layers = nn.ModuleList()
        self.edge_weights = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(self.args.graphcnng_layers) - 1):
            self.edge_weights.append(nn.Linear(self.args.graphcnng_layers[i], self.args.graphcnng_layers[i] * self.args.graphcnng_layers[i + 1]))
            self.layers.append(NNConv(self.args.graphcnng_layers[i], self.args.graphcnng_layers[i + 1], self.edge_weights[i], aggr='mean', root_weight=True, bias=True))
            self.bn_layers.append(torch_geometric.nn.BatchNorm(self.args.graphcnng_layers[i + 1]))

        self.edge_weights.append(nn.Linear(self.args.graphcnng_layers[-1], self.args.graphcnng_layers[-1] * self.args.node_feat_size))
        self.layers.append(NNConv(self.args.graphcnng_layers[-1], self.args.node_feat_size, self.edge_weights[-1], aggr='mean', root_weight=True, bias=True))
        self.bn_layers.append(torch_geometric.nn.BatchNorm(self.args.node_feat_size))

        logging.info("dense: ")
        logging.info(self.dense)

        logging.info("edge_weights: ")
        logging.info(self.edge_weights)

        logging.info("layers: ")
        logging.info(self.layers)

        logging.info("bn layers: ")
        logging.info(self.bn_layers)


    def forward(self, x, labels=None, epoch=None):
        x = F.leaky_relu(self.dense(x), negative_slope=self.args.leaky_relu_alpha)

        batch_size = x.size(0)
        x = x.reshape(batch_size * self.args.num_hits, self.args.graphcnng_layers[0])
        zeros = torch.zeros(batch_size * self.args.num_hits, dtype=int).to(self.args.device)
        zeros[torch.arange(batch_size) * self.args.num_hits] = 1
        batch = torch.cumsum(zeros, 0) - 1

        for i in range(len(self.layers)):
            edge_index = knn_graph(x, self.args.num_knn, batch)
            edge_attr = x[edge_index[0]] - x[edge_index[1]]
            x = self.bn_layers[i](self.layers[i](x, edge_index, edge_attr))
            if i < (len(self.layers) - 1): x = F.leaky_relu(x, negative_slope=self.args.leaky_relu_alpha)

        if self.args.graphcnng_tanh: x = F.tanh(x)

        return x.reshape(batch_size, self.args.num_hits, self.args.node_feat_size)
