# Getting mu and sigma of activation features of GCNN classifier for the FID score

import torch
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import numpy as np

import utils

from os import path

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from energyflow.emd import emds

import logging

from particlenet import ParticleNet


def get_mu2_sigma2(args, C, X_loaded, fullpath):
    logging.info("Getting mu2, sigma2")

    C.eval()
    for i, jet in tqdm(enumerate(X_loaded), total=len(X_loaded)):
        if(i == 0): activations = C(jet[0][:, :, :3].to(args.device), ret_activations=True).cpu().detach()
        else: activations = torch.cat((C(jet[0][:, :, :3].to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)

    activations = activations.numpy()

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    np.savetxt(fullpath + "mu2.txt", mu)
    np.savetxt(fullpath + "sigma2.txt", sigma)

    return mu, sigma


def load(args, X_loaded=None):
    C = ParticleNet(args.num_hits, args.node_feat_size, device=args.device).to(args.device)
    C.load_state_dict(torch.load(args.evaluation_path + "C_state_dict.pt", map_location=args.device))

    fullpath = args.evaluation_path + args.jets
    logging.debug(fullpath)
    if path.exists(fullpath + "mu2.txt"):
        mu2 = np.loadtxt(fullpath + "mu2.txt")
        sigma2 = np.loadtxt(fullpath + "sigma2.txt")
    else:
        mu2, sigma2 = get_mu2_sigma2(args, C, X_loaded, fullpath)

    return (C, mu2, sigma2)


rng = np.random.default_rng()


# make sure to deepcopy G passing in
def calc_jsd(args, X, G):
    logging.info("evaluating JSD")
    G.eval()

    bins = [np.arange(-1, 1, 0.02), np.arange(-1, 1, 0.02), np.arange(-1, 1, 0.01)]
    N = len(X)

    jsds = []

    for j in tqdm(range(10)):
        gen_out = utils.gen(args, G, num_samples=args.batch_size).cpu().detach().numpy()
        for i in range(int(args.num_samples / args.batch_size)):
            gen_out = np.concatenate((gen_out, utils.gen(args, G, num_samples=args.batch_size).cpu().detach().numpy()), 0)
        gen_out = gen_out[:args.num_samples]

        sample = X[rng.choice(N, size=args.num_samples, replace=False)].cpu().detach().numpy()
        jsd = []

        for i in range(3):
            hist1 = np.histogram(gen_out[:, :, i].reshape(-1), bins=bins[i], density=True)[0]
            hist2 = np.histogram(sample[:, :, i].reshape(-1), bins=bins[i], density=True)[0]
            jsd.append(jensenshannon(hist1, hist2))

        jsds.append(jsd)

    return np.mean(np.array(jsds), axis=0), np.std(np.array(jsds), axis=0)


# make sure to deepcopy G passing in
def calc_w1(args, X, G, losses, X_loaded=None, pcgan_args=None):
    logging.info("Evaluating 1-WD")

    G.eval()
    gen_out = utils.gen_multi_batch(args, G, args.eval_tot_samples, X_loaded=X_loaded, pcgan_args=pcgan_args)

    logging.info("Generated Data")

    X_rn, mask_real = utils.unnorm_data(args, X.cpu().detach().numpy()[:args.eval_tot_samples], real=True)
    gen_out_rn, mask_gen = utils.unnorm_data(args, gen_out[:args.eval_tot_samples], real=False)

    logging.info("Unnormed data")

    if args.jf:
        realjf = utils.jet_features(X_rn, mask=mask_real)

        logging.info("Obtained real jet features")

        genjf = utils.jet_features(gen_out_rn, mask=mask_gen)

        logging.info("Obtained gen jet features")

        realefp = utils.efp(args, X_rn, mask=mask_real, real=True)

        logging.info("Obtained Real EFPs")

        genefp = utils.efp(args, gen_out_rn, mask=mask_gen, real=False)

        logging.info("Obtained Gen EFPs")

    num_batches = np.array(args.eval_tot_samples / np.array(args.w1_num_samples), dtype=int)

    for k in range(len(args.w1_num_samples)):
        logging.info("Num Samples: " + str(args.w1_num_samples[k]))
        w1s = []
        if args.jf: w1js = []
        for j in range(num_batches[k]):
            G_rand_sample = rng.choice(args.eval_tot_samples, size=args.w1_num_samples[k])
            X_rand_sample = rng.choice(args.eval_tot_samples, size=args.w1_num_samples[k])

            Gsample = gen_out_rn[G_rand_sample]
            Xsample = X_rn[X_rand_sample]

            if args.mask:
                mask_gen_sample = mask_gen[G_rand_sample]
                mask_real_sample = mask_real[X_rand_sample]
                parts_real = Xsample[mask_real_sample]
                parts_gen = Gsample[mask_gen_sample]
            else:
                parts_real = Xsample.reshape(-1, args.node_feat_size)
                parts_gen = Gsample.reshape(-1, args.node_feat_size)

            if not len(parts_gen): w1 = [1, 1, 1]
            else: w1 = [wasserstein_distance(parts_real[:, i].reshape(-1), parts_gen[:, i].reshape(-1)) for i in range(3)]
            w1s.append(w1)

            if args.jf:
                realjf_sample = realjf[X_rand_sample]
                genjf_sample = genjf[G_rand_sample]

                realefp_sample = realefp[X_rand_sample]
                genefp_sample = genefp[X_rand_sample]

                w1jf = [wasserstein_distance(realjf_sample[:, i], genjf_sample[:, i]) for i in range(2)]
                w1jefp = [wasserstein_distance(realefp_sample[:, i], genefp_sample[:, i]) for i in range(5)]

                w1js.append([i for t in (w1jf, w1jefp) for i in t])
                # w1js.append(w1jf)

        losses['w1_' + str(args.w1_num_samples[k]) + 'm'].append(np.mean(np.array(w1s), axis=0))
        losses['w1_' + str(args.w1_num_samples[k]) + 'std'].append(np.std(np.array(w1s), axis=0))

        if args.jf:
            losses['w1j_' + str(args.w1_num_samples[k]) + 'm'].append(np.mean(np.array(w1js), axis=0))
            losses['w1j_' + str(args.w1_num_samples[k]) + 'std'].append(np.std(np.array(w1js), axis=0))

    return gen_out


def get_fpnd(args, C, gen_out, mu2, sigma2):
    logging.info("Evaluating FPND")

    gen_out_loaded = DataLoader(TensorDataset(torch.tensor(gen_out)), batch_size=args.fpnd_batch_size)

    logging.info("Getting ParticleNet Acivations")
    C.eval()
    for i, gen_jets in tqdm(enumerate(gen_out_loaded), total=len(gen_out_loaded)):
        gen_jets = gen_jets[0]
        if args.mask:
            mask = gen_jets[:, :, 3:4] >= 0
            gen_jets = (gen_jets * mask)[:, :, :3]
        if(i == 0): activations = C(gen_jets.to(args.device), ret_activations=True).cpu().detach()
        else: activations = torch.cat((C(gen_jets.to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)

    activations = activations.numpy()

    mu1 = np.mean(activations, axis=0)
    sigma1 = np.cov(activations, rowvar=False)

    fpnd = utils.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    logging.info("PFND: " + str(fpnd))

    return fpnd


def calc_cov_mmd(args, X, gen_out, losses, X_loaded=None):
    X_rn, mask_real = utils.unnorm_data(args, X.cpu().detach().numpy()[:args.eval_tot_samples], real=True)
    gen_out_rn, mask_gen = utils.unnorm_data(args, gen_out[:args.eval_tot_samples], real=False)

    # converting into EFP format
    X_rn = np.concatenate((np.expand_dims(X_rn[:, :, 2], 2), X_rn[:, :, :2], np.zeros((X_rn.shape[0], X_rn.shape[1], 1))), axis=2)
    gen_out_rn = np.concatenate((np.expand_dims(gen_out_rn[:, :, 2], 2), gen_out_rn[:, :, :2], np.zeros((gen_out_rn.shape[0], gen_out_rn.shape[1], 1))), axis=2)

    logging.info("Calculating coverage and MMD")
    covs = []
    mmds = []

    for j in range(args.cov_mmd_num_batches):
        G_rand_sample = rng.choice(args.eval_tot_samples, size=args.cov_mmd_num_samples)
        X_rand_sample = rng.choice(args.eval_tot_samples, size=args.cov_mmd_num_samples)

        Gsample = gen_out_rn[G_rand_sample]
        Xsample = X_rn[X_rand_sample]

        dists = emds(Gsample, Xsample)

        mmds.append(np.mean(np.min(dists, axis=0)))
        covs.append(np.unique(np.argmin(dists, axis=1)).size / args.cov_mmd_num_samples)

    losses['coverage'].append(np.mean(np.array(covs)))
    losses['mmd'].append(np.mean(np.array(mmds)))
