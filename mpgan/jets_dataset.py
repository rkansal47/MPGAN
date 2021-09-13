import torch
from torch.utils.data import Dataset
import logging
from os.path import exists


class JetsDataset(Dataset):
    def __init__(self, args, train=True):
        pt_file = args.datasets_path + args.jets + "_jets.pt"
        logging.info(f"{pt_file = }")

        if not exists(pt_file):
            self.download_and_convert_to_pt(args.datasets_path, args.jets)

        dataset = torch.load(args.datasets_path + args.jets + "_jets.pt").float()[:, : args.num_hits, :]
        if not args.mask:
            dataset = dataset[:, :, : args.node_feat_size]

        if args.coords == "cartesian":
            args.maxp = float(torch.max(torch.abs(dataset)))
            dataset = dataset / args.maxp

            cutoff = int(dataset.size(0) * args.ttsplit)

            if args.train:
                self.X = dataset[:cutoff]
            else:
                self.X = dataset[cutoff:]
        elif args.coords == "polarrel" or args.coords == "polarrelabspt":
            args.maxepp = [float(torch.max(torch.abs(dataset[:, :, i]))) for i in range(3)]
            if hasattr(args, "mask_feat") and args.mask_feat:
                args.maxepp.append(1.0)

            logging.debug("Max Vals: " + str(args.maxepp))
            for i in range(3):
                dataset[:, :, i] /= args.maxepp[i]

            dataset[:, :, 2] -= 0.5  # pT is normalized between -0.5 and 0.5 (instead of ±1) so the peak pT lies in linear region of tanh
            if args.dataset == "jets-lagan":
                dataset[:, :, 3] -= 0.5
            dataset *= args.norm
            self.X = dataset
            args.pt_cutoff = torch.unique(self.X[:, :, 2], sorted=True)[1]  # smallest particle pT after 0
            logging.debug("Cutoff: " + str(args.pt_cutoff))

        if hasattr(args, "mask_c") and args.mask_c:
            num_particles = (torch.sum(dataset[:, :, 3] + 0.5, dim=1) / args.num_hits).unsqueeze(1)
            logging.debug("num particles: " + str(torch.sum(dataset[:, :, 3] + 0.5, dim=1)))

            if args.clabels:
                self.jet_features = torch.cat((self.jet_features, num_particles), dim=1)
            else:
                self.jet_features = num_particles
        else:
            self.jet_features = torch.zeros((len(dataset)), 1)

        if hasattr(args, "noise_padding") and args.noise_padding:
            logging.debug("pre-noise padded dataset: \n {}".format(dataset[:2, -10:]))

            noise_padding = torch.randn((len(dataset), args.num_hits, 3)) / 6

            # DOUBLE CHECK
            # noise_padding[:, :, 2] = torch.relu(noise_padding[:, :, 2])
            # noise_padding[:, :, 2] -= 0.5
            # # noise_padding[noise_padding[:, :, 2] < -0.5][:, :, 2] = -0.5

            noise_padding[:, :, 2] += 0.5
            mask = (dataset[:, :, 3] + 0.5).bool()
            noise_padding[mask] = 0
            dataset += torch.cat((noise_padding, torch.zeros((len(dataset), args.num_hits, 1))), dim=2)

            logging.debug("noise padded dataset: \n {}".format(dataset[:2, -10:]))

        tcut = int(len(self.X) * args.ttsplit)
        self.X = self.X[:tcut] if train else self.X[tcut:]
        if self.jet_features is not None:
            self.jet_features = self.jet_features[:tcut] if train else self.jet_features[tcut:]
        logging.info("Dataset shape: " + str(self.X.shape))

        logging.debug("Dataset \n" + str(self.X[:2, :10]))

    def download_and_convert_to_pt(self, data_dir: str, jet_type: str):
        """Download jet dataset and convert and save to pytorch tensor"""
        csv_file = f"{data_dir}/{jet_type}_jets.csv"
        logging.info(f"{csv_file = }")

        if not exists(csv_file):
            logging.info(f"Downloading {jet_type} jets csv")
            self.download(jet_type, csv_file)

        logging.info(f"Converting {jet_type} jets csv to pt")
        self.csv_to_pt(data_dir, jet_type, csv_file)

    def download(self, jet_type: str, csv_file: str):
        """Downloads the `jet_type` jet csv from Zenodo and saves it as `csv_file`"""
        import requests
        import sys

        records_url = "https://zenodo.org/api/records/5502543"
        r = requests.get(records_url).json()
        key = f"{jet_type}_jets.csv"
        file_url = next(item for item in r['files'] if item["key"] == key)['links']['self']  # finding the url for the particular jet type dataset
        logging.info(f"Downloading {file_url = }")

        # from https://sumit-ghosh.com/articles/python-download-progress-bar/
        with open(csv_file, "wb") as f:
            response = requests.get(file_url, stream=True)
            total = response.headers.get("content-length")

            if total is None:
                f.write(response.content)
            else:
                downloaded = 0
                total = int(total)

                for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * downloaded / total)
                    sys.stdout.write("\r[{}{}] {:.0f}%".format("█" * done, "." * (50 - done), float(downloaded / total) * 100))
                    sys.stdout.flush()

        sys.stdout.write("\n")

    def csv_to_pt(self, data_dir: str, jet_type: str, csv_file: str):
        """Converts and saves downloaded csv file to pytorch tensor"""
        import numpy as np

        pt_file = f"{data_dir}/{jet_type}_jets.pt"
        torch.save(torch.tensor(np.loadtxt(csv_file).reshape(-1, 30, 4)), pt_file)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.jet_features[idx]


class JetsClassifierDataset(Dataset):
    def __init__(self, args, train=True, lim=None):
        mask = "_mask" if args.mask else ""

        jet_types = ["g", "t", "q", "w", "z"]
        classidx = list(range(5))

        Xtemp = torch.load(args.datasets_path + "all_" + jet_types[0] + "_jets_150p_polarrel" + mask + ".pt").float()[:, : args.num_hits, :]
        tcut = int(len(Xtemp) * 0.7)
        X = Xtemp[:tcut] if train else Xtemp[tcut:]
        if lim is not None:
            X = X[:lim]
        Y = torch.zeros(len(X), dtype=int) + classidx[0]

        for jet_type, idx in zip(jet_types[1:], classidx[1:]):
            Xtemp = torch.load(args.datasets_path + "all_" + jet_type + "_jets_150p_polarrel" + mask + ".pt").float()[:, : args.num_hits, :]
            tcut = int(len(Xtemp) * 0.7)
            Xtemp = Xtemp[:tcut] if train else Xtemp[tcut:]
            if lim is not None:
                Xtemp = Xtemp[:lim]
            X = torch.cat((X, Xtemp), dim=0)
            Y = torch.cat((Y, torch.zeros(len(Xtemp), dtype=int) + idx), dim=0)

        args.maxepp = [float(torch.max(torch.abs(X[:, :, i]))) for i in range(3)]
        if args.mask:
            args.maxepp.append(1.0)

        logging.debug("Max Vals: " + str(args.maxepp))
        for i in range(args.node_feat_size):
            X[:, :, i] /= args.maxepp[i]

        X[:, :, 2] -= 0.5  # pT is normalized between -0.5 and 0.5 so the peak pT lies in linear region of tanh
        # dataset *= args.norm
        self.X = X
        self.Y = Y

        logging.info("X shape: " + str(self.X.shape))
        logging.info("Y shape: " + str(self.Y.shape))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
