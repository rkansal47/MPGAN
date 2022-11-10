"""
Collection of methods for calculating distances and divergences between two distributions.

Author: Raghav Kansal
"""


from typing import Callable
from numpy.typing import ArrayLike

import numpy as np
from scipy import linalg
from scipy.stats import iqr
from scipy.optimize import curve_fit

from concurrent.futures import ThreadPoolExecutor

import logging

from numpy import njit, prange


def normalise_features(X: ArrayLike, Y: ArrayLike = None):
    maxes = np.max(np.abs(X), axis=0)

    return (X / maxes, Y / maxes) if Y is not None else X / maxes


def linear(x, intercept, slope):
    return intercept + slope * x


def _average_batches(X, Y, batch_size, num_batches, seed):
    np.random.seed(seed)

    vals_point = []
    for _ in range(num_batches):
        rand1 = np.random.choice(len(X), size=batch_size)
        rand2 = np.random.choice(len(Y), size=batch_size)

        rand_sample1 = X[rand1]
        rand_sample2 = Y[rand2]

        val = frechet_gaussian_distance(rand_sample1, rand_sample2, normalise=False)
        vals_point.append(val)

    return [np.mean(vals_point), np.std(vals_point)]


# based on https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py
def fpd_infinity(
    X: ArrayLike,
    Y: ArrayLike,
    min_samples: int = 5_000,
    max_samples: int = 50_000,
    num_batches: int = 10,
    num_points: int = 200,
    seed: int = 42,
    normalise: bool = True,
    n_jobs: int = 1,
):
    if normalise:
        X, Y = normalise_features(X, Y)

    # Choose the number of images to evaluate FID_N at regular intervals over N
    batches = (1 / np.linspace(1.0 / min_samples, 1.0 / max_samples, num_points)).astype("int32")
    # batches = np.linspace(min_samples, max_samples, num_points).astype("int32")

    np.random.seed(seed)

    vals = []

    for i, batch_size in enumerate(batches):
        vals_point = []
        for _ in range(num_batches):
            rand1 = np.random.choice(len(X), size=batch_size)
            rand2 = np.random.choice(len(Y), size=batch_size)

            rand_sample1 = X[rand1]
            rand_sample2 = Y[rand2]

            val = frechet_gaussian_distance(rand_sample1, rand_sample2, normalise=False)
            vals_point.append(val)

        vals.append(np.mean(vals_point))

    vals = np.array(vals)

    params, covs = curve_fit(linear, 1 / batches, vals, bounds=([0, 0], [np.inf, np.inf]))

    return (params[0], np.sqrt(np.diag(covs)[0]))


# from https://github.com/mseitzer/pytorch-fid
def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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
            f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        )
        logging.debug(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def frechet_gaussian_distance(X: ArrayLike, Y: ArrayLike, normalise: bool = True) -> float:
    if normalise:
        X, Y = normalise_features(X, Y)

    mu1 = np.mean(X, axis=0)
    sigma1 = np.cov(X, rowvar=False)
    mu2 = np.mean(Y, axis=0)
    sigma2 = np.cov(Y, rowvar=False)

    return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


@njit
def _mmd_quadratic_unbiased(XX: ArrayLike, YY: ArrayLike, XY: ArrayLike):
    m, n = XX.shape[0], YY.shape[0]
    # subtract diagonal 1s
    return (
        (XX.sum() - np.trace(XX)) / (m * (m - 1))
        + (YY.sum() - np.trace(YY)) / (n * (n - 1))
        - 2 * np.mean(XY)
    )


def _get_mmd_quadratic_arrays(X: ArrayLike, Y: ArrayLike, kernel_func: Callable, **kernel_args):
    XX = kernel_func(X, X, **kernel_args)
    YY = kernel_func(Y, Y, **kernel_args)
    XY = kernel_func(X, Y, **kernel_args)
    return XX, YY, XY


@njit
def _poly_kernel_pairwise(X: ArrayLike, Y: ArrayLike, degree: int) -> np.ndarray:
    gamma = 1.0 / X.shape[-1]
    return (X @ Y.T * gamma + 1.0) ** degree


@njit
def mmd_poly_quadratic_unbiased(X: ArrayLike, Y: ArrayLike, degree: int = 4) -> float:
    XX = _poly_kernel_pairwise(X, X, degree=degree)
    YY = _poly_kernel_pairwise(Y, Y, degree=degree)
    XY = _poly_kernel_pairwise(X, Y, degree=degree)
    return _mmd_quadratic_unbiased(XX, YY, XY)


def multi_batch_evaluation(
    X: ArrayLike,
    Y: ArrayLike,
    num_batches: int,
    batch_size: int,
    metric: Callable,
    seed: int = 42,
    normalise: bool = True,
    **metric_args,
):
    np.random.seed(seed)

    vals = []
    for _ in range(num_batches):
        rand1 = np.random.choice(len(X), size=batch_size)
        rand2 = np.random.choice(len(Y), size=batch_size)

        rand_sample1 = X[rand1]
        rand_sample2 = Y[rand2]

        val = metric(rand_sample1, rand_sample2, normalise=normalise, **metric_args)
        vals.append(val)

    mean_std = (np.mean(vals, axis=0), np.std(vals, axis=0))

    return mean_std


@njit(parallel=True)
def _average_batches_mmd(X, Y, num_batches, batch_size, seed):
    vals_point = []
    for i in prange(num_batches):
        np.random.seed(seed + i * 1000)  # in case of multi-threading
        rand1 = np.random.choice(len(X), size=batch_size)
        rand2 = np.random.choice(len(Y), size=batch_size)

        rand_sample1 = X[rand1]
        rand_sample2 = Y[rand2]

        val = mmd_poly_quadratic_unbiased(rand_sample1, rand_sample2, normalise=False, degree=4)
        vals_point.append(val)

    return vals_point


def multi_batch_evaluation_mmd(
    X: ArrayLike,
    Y: ArrayLike,
    num_batches: int = 10,
    batch_size: int = 5000,
    seed: int = 42,
    normalise: bool = True,
):
    if normalise:
        X, Y = normalise_features(X, Y)

    vals_point = _average_batches_mmd(X, Y, num_batches, batch_size, seed)
    return [np.median(vals_point), iqr(vals_point, rng=(16.275, 83.725))]
