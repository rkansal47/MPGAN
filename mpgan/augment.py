import torch
import numpy as np


def rand_mix(args, X1, X2, p):
    if p == 1:
        return X1

    assert X1.size(0) == X2.size(0), "Error: different batch sizes of rand mix data"
    batch_size = X1.size(0)

    rand = torch.rand(batch_size, 1, 1).to(args.device)
    mix = torch.zeros(batch_size, 1, 1).to(args.device)
    mix[rand < p] = 1

    return X1 * (1 - mix) + X2 * mix


def augment(args, X, p):
    if args.aug_r90:
        X = rand_mix(args, X, rand_90_rotation(args, X), p)
    if args.aug_f:
        X = rand_mix(args, X, rand_flip(args, X), p)
    if args.aug_t:
        X = rand_mix(args, X, rand_translate(args, X), p)
    if args.aug_s:
        X = rand_mix(args, X, rand_scale(args, X), p)

    return X


def rand_flip(args, X):
    batch_size = X.size(0)
    flip_xy = ((torch.round(torch.rand(batch_size, 1, 2).to(args.device)) * 2) - 1).repeat(
        1, args.num_hits, 1
    )
    ones = torch.ones(batch_size, args.num_hits, 1).to(args.device)
    mult = torch.cat((flip_xy, ones), axis=2)
    return X * mult


def rand_90_rotation(args, X):
    batch_size = X.size(0)

    angle = torch.floor(torch.rand(batch_size, 1, 1).to(args.device) * 4) * (np.pi / 2)
    sin = torch.sin(angle)
    cos = torch.cos(angle)

    zeros = torch.zeros(batch_size, 1, 1).to(args.device)
    ones = torch.ones(batch_size, 1, 1).to(args.device)

    rot_mat = torch.cat(
        (
            torch.cat((cos, -sin, zeros), axis=2),
            torch.cat((sin, cos, zeros), axis=2),
            torch.cat((zeros, zeros, ones), axis=2),
        ),
        axis=1,
    )

    return torch.matmul(rot_mat.unsqueeze(1), X.unsqueeze(3)).squeeze(3)


def rand_translate(args, X):
    batch_size = X.size(0)
    shift_xy = (
        (torch.rand(batch_size, 1, 2).to(args.device) - 0.5).repeat(1, args.num_hits, 1)
    ) * args.translate_ratio
    zeros = torch.zeros(batch_size, args.num_hits, 1).to(args.device)
    add = torch.cat((shift_xy, zeros), axis=2)
    return X + add


def rand_translate_per_node(args, X):
    batch_size = X.size(0)
    shift_xy = (
        torch.rand(batch_size, args.num_hits, 2).to(args.device) - 0.5
    ) * args.translate_pn_ratio
    zeros = torch.zeros(batch_size, args.num_hits, 1).to(args.device)
    add = torch.cat((shift_xy, zeros), axis=2)
    return X + add


def rand_scale(args, X):
    batch_size = X.size(0)
    lognormal = torch.distributions.log_normal.LogNormal(
        torch.tensor([0.0]).to(args.device),
        torch.tensor([args.scale_sd]).to(args.device),
    )
    scale = lognormal.sample((batch_size, 1)).repeat(1, args.num_hits, 2)
    ones = torch.ones(batch_size, args.num_hits, 1).to(args.device)
    mult = torch.cat((scale, ones), axis=2)
    return X * mult
