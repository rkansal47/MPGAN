from typing import List, Union, Optional

import torch
from torch import Tensor
import numpy as np

import logging

# from os.path import exists


class CaloGANDataset(torch.utils.data.Dataset):
    _num_non_mask_features = 4

    def __init__(
        self,
        data_dir: str = "./datasets/",
        download: bool = False,
        num_particles: int = 30,
        normalize: bool = True,
        feature_norms: List[float] = [None, None, None, 1.0, 1.0],
        feature_shifts: List[float] = [-0.5, -0.5, -0.5, -0.5, -0.5],
        logE: bool = False,
        use_mask: bool = True,
        train: bool = True,
        train_fraction: float = 0.7,
    ):
        self.feature_norms = feature_norms
        self.feature_shifts = feature_shifts
        self.use_mask = use_mask
        self.normalize = normalize
        self.num_particles = num_particles
        self.logE = logE

        npy_file = f"{data_dir}/calogan_graph_30_hits_etaphizEmask.npy"

        dataset = Tensor(np.load(npy_file))

        if self.logE:
            dataset[:, :, 3] = torch.log(dataset[:, :, 3] + 1e-12)
            self.feature_shifts[3] = 0

        jet_features = self.get_jet_features(dataset)

        logging.info(f"Loaded dataset {dataset.shape = }")
        if normalize:
            logging.info("Normalizing features")
            self.feature_maxes = self.normalize_features(dataset, feature_norms, feature_shifts)

        tcut = int(len(dataset) * train_fraction)

        self.data = dataset[:tcut] if train else dataset[tcut:]
        self.jet_features = jet_features[:tcut] if train else jet_features[tcut:]

        print(self.data[0, :10])

        logging.info("Dataset processed")

    def get_jet_features(self, dataset: Tensor) -> Tensor:
        """
        Returns jet-level features. `Will be expanded to include jet pT and eta.`

        Args:
            dataset (Tensor):  dataset tensor of shape [N, num_particles, num_features],
              where the last feature is the mask.
            use_num_particles_jet_feature (bool): `Currently does nothing,
              in the future such bools will specify which jet features to use`.

        Returns:
            Tensor: jet features tensor of shape [N, num_jet_features].

        """
        jet_num_particles = (torch.sum(dataset[:, :, -1], dim=1) / self.num_particles).unsqueeze(1)
        logging.debug("{num_particles = }")
        return jet_num_particles

    @classmethod
    def normalize_features(
        self,
        dataset: Tensor,
        feature_norms: Union[float, List[float]] = 1.0,
        feature_shifts: Union[float, List[float]] = 0.0,
    ) -> Optional[List]:
        """
        Normalizes dataset features (in place),
        by scaling to ``feature_norms`` maximum and shifting by ``feature_shifts``.

        If the value in the List for a feature is None, it won't be scaled or shifted.

        If ``fpnd`` is True, will normalize instead to the same scale as was used for the
        ParticleNet training in https://arxiv.org/abs/2106.11535.

        Args:
            dataset (Tensor): dataset tensor of shape [N, num_particles, num_features].
            feature_norms (Union[float, List[float]]): max value to scale each feature to.
              Can either be a single float for all features, or a list of length ``num_features``.
              Defaults to 1.0.
            feature_shifts (Union[float, List[float]]): after scaling, value to shift feature by.
              Can either be a single float for all features, or a list of length ``num_features``.
              Defaults to 0.0.
            fpnd (bool): Normalize features for ParticleNet inference for the
              Frechet ParticleNet Distance metric. Will override `feature_norms`` and
              ``feature_shifts`` inputs. Defaults to False.

        Returns:
            Optional[List]: if ``fpnd`` is False, returns list of length ``num_features``
            of max absolute values for each feature. Used for unnormalizing features.

        """
        num_features = dataset.shape[2]

        feature_maxes = [float(torch.max(torch.abs(dataset[:, :, i]))) for i in range(num_features)]

        if isinstance(feature_norms, float):
            feature_norms = np.full(num_features, feature_norms)

        if isinstance(feature_shifts, float):
            feature_shifts = np.full(num_features, feature_shifts)

        logging.debug(f"{feature_maxes = }")

        for i in range(num_features):
            if feature_norms[i] is not None:
                dataset[:, :, i] /= feature_maxes[i]
                dataset[:, :, i] *= feature_norms[i]

            if feature_shifts[i] is not None and feature_shifts[i] != 0:
                dataset[:, :, i] += feature_shifts[i]

        return feature_maxes

    def unnormalize_features(
        self,
        dataset: Union[Tensor, np.ndarray],
        ret_mask_separate: bool = True,
        is_real_data: bool = False,
        zero_mask_particles: bool = True,
        zero_neg_pt: bool = True,
    ):
        """
        Inverts the ``normalize_features()`` function on the input ``dataset`` array or tensor,
        plus optionally zero's the masked particles and negative pTs.
        Only applicable if dataset was normalized first
        i.e. ``normalize`` arg into JetNet instance is True.

        Args:
            dataset (Union[Tensor, np.ndarray]): Dataset to unnormalize.
            ret_mask_separate (bool): Return the jet and mask separately. Defaults to True.
            is_real_data (bool): Real or generated data. Defaults to False.
            zero_mask_particles (bool): Set features of zero-masked particles to 0.
              Not needed for real data. Defaults to True.
            zero_neg_pt (bool): Set pT to 0 for particles with negative pt.
              Not needed for real data. Defaults to True.

        Returns:
            Unnormalized dataset of same type as input. Either a tensor/array of shape
            ``[num_jets, num_particles, num_features (including mask)]`` if ``ret_mask_separate``
            is False, else a tuple with a tensor/array of shape
            ``[num_jets, num_particles, num_features (excluding mask)]`` and another binary mask
            tensor/array of shape ``[num_jets, num_particles, 1]``.
        """
        if not self.normalize:
            raise RuntimeError("Can't unnormalize features if dataset has not been normalized.")

        num_features = dataset.shape[2]

        for i in range(num_features):
            if self.feature_shifts[i] is not None and self.feature_shifts[i] != 0:
                dataset[:, :, i] -= self.feature_shifts[i]

            if self.feature_norms[i] is not None:
                dataset[:, :, i] /= self.feature_norms[i]
                dataset[:, :, i] *= self.feature_maxes[i]

        if self.logE:
            dataset[:, :, 3] = torch.exp(dataset[:, :, 3])

        mask = dataset[:, :, -1] >= 0.5 if self.use_mask else None

        if not is_real_data and zero_mask_particles and self.use_mask:
            dataset[~mask] = 0

        if not is_real_data and zero_neg_pt:
            dataset[:, :, 2][dataset[:, :, 2] < 0] = 0

        return dataset[:, :, : self._num_non_mask_features], mask if ret_mask_separate else dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.jet_features[idx]
