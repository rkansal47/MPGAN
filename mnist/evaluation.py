# Getting mu and sigma of activation features of GCNN classifier for the FID score

import torch
import torch.nn.functional as F
from torch import Tensor

from torch.utils.data import DataLoader

import torch_geometric.transforms as T

from torch_geometric.utils import normalized_cut
from torch_geometric.nn import graclus, max_pool, global_mean_pool
from torch_geometric.nn import GMMConv
from torch_geometric.data import Batch, Data

from tqdm import tqdm

import numpy as np
from scipy import linalg

from os import path
import pathlib

import logging

cutoff = 0.32178
FID_EVAL_SIZE = 8192

# transform my format to torch_geometric's
def tg_transform(X: Tensor, num_hits: int, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = X.size(0)

    pos = X[:, :, :2]

    x1 = pos.repeat(1, 1, num_hits).reshape(batch_size, num_hits * num_hits, 2)
    x2 = pos.repeat(1, num_hits, 1)

    diff_norms = torch.norm(x2 - x1 + 1e-12, dim=2)

    # diff = x2-x1
    # diff = diff[diff_norms < args.cutoff]

    norms = diff_norms.reshape(batch_size, num_hits, num_hits)
    neighborhood = torch.nonzero(norms < cutoff, as_tuple=False)
    # diff = diff[neighborhood[:, 1] != neighborhood[:, 2]]

    neighborhood = neighborhood[neighborhood[:, 1] != neighborhood[:, 2]]  # remove self-loops
    unique, counts = torch.unique(neighborhood[:, 0], return_counts=True)
    # edge_slices = torch.cat((torch.tensor([0]).to(device), counts.cumsum(0)))
    edge_index = (neighborhood[:, 1:] + (neighborhood[:, 0] * num_hits).view(-1, 1)).transpose(0, 1)

    x = X[:, :, 2].reshape(batch_size * num_hits, 1) + 0.5
    pos = 28 * pos.reshape(batch_size * num_hits, 2) + 14

    row, col = edge_index
    edge_attr = (pos[col] - pos[row]) / (2 * 28 * cutoff) + 0.5

    zeros = torch.zeros(batch_size * num_hits, dtype=int).to(device)
    zeros[torch.arange(batch_size) * num_hits] = 1
    batch = torch.cumsum(zeros, 0) - 1

    return Batch(
        batch=batch, x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=None, pos=pos
    )


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class MoNet(torch.nn.Module):
    def __init__(self, kernel_size):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(1, 32, dim=2, kernel_size=kernel_size)
        self.conv2 = GMMConv(32, 64, dim=2, kernel_size=kernel_size)
        self.conv3 = GMMConv(64, 64, dim=2, kernel_size=kernel_size)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, data):
        row, col = data.edge_index
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * 28 * cutoff) + 0.5

        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        row, col = data.edge_index
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * 28 * cutoff) + 0.5

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        row, col = data.edge_index
        data.edge_attr = (data.pos[col] - data.pos[row]) / (2 * 28 * cutoff) + 0.5

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = global_mean_pool(data.x, data.batch)
        return self.fc1(x)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


def get_mu2_sigma2(args, C, X_loaded, fullpath):
    print("getting mu2, sigma2")
    activations = 0
    for batch_ndx, data in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        tg_data = tg_transform(args, data.to(args.device))
        if batch_ndx % args.gpu_batch == 0:
            if batch_ndx == args.gpu_batch:
                np_activations = activations.cpu().detach().numpy()
            elif batch_ndx > args.gpu_batch:
                np_activations = np.concatenate(
                    (np_activations, activations.cpu().detach().numpy())
                )
            activations = C(tg_data)
        else:
            activations = torch.cat((C(tg_data), activations), axis=0)
        # if batch_ndx == 113:
        #     break

    activations = np.concatenate(
        (np_activations, activations.cpu().detach().numpy())
    )  # because torch doesn't have a built in function for calculating the covariance matrix

    print(activations.shape)

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    np.savetxt(fullpath + "mu2.txt", mu)
    np.savetxt(fullpath + "sigma2.txt", sigma)

    return mu, sigma


def load(
    num_hits: int,
    eval_path: str,
    num: int,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    C = MoNet(25).to(device)
    mstr = "C_sm_nh_" + str(num_hits)

    sd = torch.load(eval_path + mstr + "_state_dict.pt")
    # update keys to new PyG format
    for i in range(1, 4):
        sd[f"conv{i}.root.weight"] = sd[f"conv{i}.root"].T

    C.load_state_dict(sd)
    numstr = str(num) if num != -1 else "all_nums"
    dstr = "_sm_2_nh_" + str(num_hits) + "_"
    fullpath = eval_path + numstr + dstr
    logging.info(f"Loading mu2 sigma 2 from {fullpath}")
    if path.exists(fullpath + "mu2.txt"):
        mu2 = np.loadtxt(fullpath + "mu2.txt")
        sigma2 = np.loadtxt(fullpath + "sigma2.txt")
    else:
        raise RuntimeError("mu2 sigma2 path doesn't exist")
        # mu2, sigma2 = get_mu2_sigma2(args, C, X_loaded, fullpath)
    return (C, mu2, sigma2)


# from https://github.com/mseitzer/pytorch-fid
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; " "adding %s to diagonal of cov estimates"
        ) % eps
        logging.info(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# make sure to deepcopy G passing in
def get_fid(X: np.ndarray, num_hits: int, num: int, batch_size: int):
    eval_path = str(pathlib.Path(__file__).parent.resolve()) + "/evaluation_resources/"
    C, mu2, sigma2 = load(num_hits, eval_path, num)
    logging.info("evaluating fid")
    C.eval()

    # run inference and store activations
    X_loaded = DataLoader(X[:FID_EVAL_SIZE], batch_size)

    logging.info(f"Calculating MoNet activations with batch size: {batch_size}")
    activations = []
    for i, jets_batch in tqdm(enumerate(X_loaded), total=len(X_loaded), desc="Running MoNet"):
        activations.append(C(tg_transform(jets_batch, num_hits)).cpu().detach().numpy())

    activations = np.concatenate(activations, axis=0)
    mu1 = np.mean(activations, axis=0)
    sigma1 = np.cov(activations, rowvar=False)

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    logging.info("fid:" + str(fid))

    return fid
