import torch
from torch.utils.data import Dataset
import numpy as np


class MNISTGraphDataset(Dataset):
    def __init__(self, data_dir, num_thresholded, train=True, intensities=True, num=-1):
        if train:
            dataset_tr = np.loadtxt(data_dir + "/mnist_train.csv", delimiter=",", dtype=np.float32)
            dataset_te = np.loadtxt(data_dir + "/mnist_test.csv", delimiter=",", dtype=np.float32)
            dataset = np.concatenate((dataset_tr, dataset_te), axis=0)
        else:
            dataset = np.loadtxt(data_dir + "mnist_test.csv", delimiter=",", dtype=np.float32)

        print("MNIST CSV Loaded")

        if isinstance(num, list):
            map1 = list(map(lambda x: x in num, dataset[:, 0]))
            dataset = dataset[map1]
        elif num > -1:
            dataset = dataset[dataset[:, 0] == num]

        print(dataset.shape)

        X_pre = (dataset[:, 1:] - 127.5) / 255.0

        imrange = np.linspace(-0.5, 0.5, num=28, endpoint=False)

        xs, ys = np.meshgrid(imrange, imrange)

        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        self.X = np.array(list(map(lambda x: np.array([xs, ys, x]).T, X_pre)))

        if not intensities:
            self.X = np.array(
                list(map(lambda x: x[x[:, 2].argsort()][-num_thresholded:, :2], self.X))
            )
        else:
            self.X = np.array(list(map(lambda x: x[x[:, 2].argsort()][-num_thresholded:], self.X)))

        self.X = torch.FloatTensor(self.X)

        print(self.X.shape)
        # print(self.X[0])
        print("Data Processed")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
