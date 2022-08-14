import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

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

        layers.append(
            nn.Linear(self.args.rgang_fc[-1], self.args.num_hits * self.args.node_feat_size)
        )
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

        self.dense = nn.Linear(
            self.args.latent_dim, self.args.num_hits * self.args.graphcnng_layers[0]
        )

        self.layers = nn.ModuleList()
        self.edge_weights = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(self.args.graphcnng_layers) - 1):
            self.edge_weights.append(
                nn.Linear(
                    self.args.graphcnng_layers[i],
                    self.args.graphcnng_layers[i] * self.args.graphcnng_layers[i + 1],
                )
            )
            self.layers.append(
                NNConv(
                    self.args.graphcnng_layers[i],
                    self.args.graphcnng_layers[i + 1],
                    self.edge_weights[i],
                    aggr="mean",
                    root_weight=True,
                    bias=True,
                )
            )
            self.bn_layers.append(torch_geometric.nn.BatchNorm(self.args.graphcnng_layers[i + 1]))

        self.edge_weights.append(
            nn.Linear(
                self.args.graphcnng_layers[-1],
                self.args.graphcnng_layers[-1] * self.args.node_feat_size,
            )
        )
        self.layers.append(
            NNConv(
                self.args.graphcnng_layers[-1],
                self.args.node_feat_size,
                self.edge_weights[-1],
                aggr="mean",
                root_weight=True,
                bias=True,
            )
        )
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
            if i < (len(self.layers) - 1):
                x = F.leaky_relu(x, negative_slope=self.args.leaky_relu_alpha)

        if self.args.graphcnng_tanh:
            x = F.tanh(x)

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
            layers.append(
                nn.Linear(self.args.pointnetd_pointfc[i], self.args.pointnetd_pointfc[i + 1])
            )
            layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))

        self.pointfc = nn.Sequential(*layers)

        layers = []

        for i in range(len(self.args.pointnetd_fc) - 1):
            layers.append(nn.Linear(self.args.pointnetd_fc[i], self.args.pointnetd_fc[i + 1]))
            if i < len(self.args.pointnetd_fc) - 2:
                layers.append(nn.LeakyReLU(negative_slope=args.leaky_relu_alpha))

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
        x = self.pointfc(x.view(batch_size * self.args.num_hits, self.args.node_feat_size)).view(
            batch_size, self.args.num_hits, self.args.pointnetd_pointfc[-1]
        )
        x = torch.cat((torch.max(x, dim=1)[0], torch.mean(x, dim=1)), dim=1)
        return self.fc(x)


# https://github.com/jtpils/TreeGAN/blob/master/layers/gcn.py
class TreeGCN(nn.Module):
    def __init__(
        self, depth, features, degrees, support=10, node=1, upsample=False, activation=True
    ):
        self.depth = depth
        self.in_feature = features[depth]
        self.out_feature = features[depth + 1]
        self.node = node
        self.degree = degrees[depth]
        self.upsample = upsample
        self.activation = activation
        super(TreeGCN, self).__init__()

        self.W_root = nn.ModuleList(
            [
                nn.Linear(features[inx], self.out_feature, bias=False)
                for inx in range(self.depth + 1)
            ]
        )

        if self.upsample:
            self.W_branch = nn.Parameter(
                torch.FloatTensor(self.node, self.in_feature, self.degree * self.in_feature)
            )

        self.W_loop = nn.Sequential(
            nn.Linear(self.in_feature, self.in_feature * support, bias=False),
            nn.Linear(self.in_feature * support, self.out_feature, bias=False),
        )

        self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.init_param()

    def init_param(self):
        if self.upsample:
            init.xavier_uniform_(self.W_branch.data, gain=init.calculate_gain("relu"))

        stdv = 1.0 / math.sqrt(self.out_feature)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, tree):
        batch = tree[0].size(0)
        root = 0
        for inx in range(self.depth + 1):
            root_num = tree[inx].size(1)
            repeat_num = int(self.node / root_num)
            root_node = self.W_root[inx](tree[inx])
            root = root + root_node.repeat(1, 1, repeat_num).view(batch, -1, self.out_feature)

        branch = 0

        if self.upsample:
            branch = tree[-1].unsqueeze(2) @ self.W_branch
            branch = self.leaky_relu(branch)
            branch = branch.view(batch, self.node * self.degree, self.in_feature)

            branch = self.W_loop(branch)

            branch = root.repeat(1, 1, self.degree).view(batch, -1, self.out_feature) + branch
        else:
            branch = self.W_loop(tree[-1])

            branch = root + branch

        if self.activation:
            branch = self.leaky_relu(branch + self.bias.repeat(1, self.node, 1))
        tree.append(branch)

        return tree


# https://github.com/jtpils/TreeGAN/blob/master/model/gan_network.py
class TreeGANG(nn.Module):
    def __init__(self, features, degrees, support):
        self.layer_num = len(features) - 1
        assert self.layer_num == len(
            degrees
        ), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(TreeGANG, self).__init__()

        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num - 1:
                self.gcn.add_module(
                    "TreeGCN_" + str(inx),
                    TreeGCN(
                        inx,
                        features,
                        degrees,
                        support=support,
                        node=vertex_num,
                        upsample=True,
                        activation=False,
                    ),
                )
            else:
                self.gcn.add_module(
                    "TreeGCN_" + str(inx),
                    TreeGCN(
                        inx,
                        features,
                        degrees,
                        support=support,
                        node=vertex_num,
                        upsample=True,
                        activation=True,
                    ),
                )
            vertex_num = int(vertex_num * degrees[inx])

        logging.info(f"TreeGANG gcn: {self.gcn}")

    def forward(self, tree, labels=None):
        feat = self.gcn(tree)

        self.pointcloud = feat[-1]

        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1]
