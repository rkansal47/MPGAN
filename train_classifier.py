import logging

from jetnet.datasets import JetNet
from ext_models.particlenet import ParticleNet

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.special import softmax

from tqdm import tqdm

import os
from os import listdir, mkdir, remove
from os.path import dirname, realpath

from setup_training import init_logging

plt.switch_backend("agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.cuda.set_device(0)
torch.manual_seed(4)
torch.autograd.set_detect_anomaly(True)


class TupleDataset(Dataset):
    def __init__(self, X, y):
        super(TupleDataset, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)


def add_bool_arg(parser, name, help, default=False, no_name=None):
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})


class objectview(object):
    """converts a dict into an object"""

    def __init__(self, d):
        self.__dict__ = d


def parse_args():
    import argparse

    dir_path = dirname(realpath(__file__))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help='log file name - default is name of file in outs/ ; "stdout" prints to console',
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    parser.add_argument(
        "--dir-path",
        type=str,
        default=dir_path,
        help="path where dataset and output will be stored",
    )
    add_bool_arg(parser, "n", "run on nautilus cluster", default=False)

    add_bool_arg(parser, "load-model", "load a pretrained model", default=False)
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="which epoch to start training on (only makes sense if loading a model)",
    )

    parser.add_argument(
        "--jets",
        type=str,
        default="g",
        help="jet type",
        choices=["g", "t", "q"],
    )

    parser.add_argument("--num_hits", type=int, default=30, help="num nodes in graph")
    parser.add_argument("--node-feat-size", type=int, default=3, help="num features per node")

    parser.add_argument("--num-epochs", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=384, help="batch size")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="pick optimizer",
        choices=["adam", "rmsprop", "adamw"],
    )
    parser.add_argument("--lr", type=float, default=3e-4)

    add_bool_arg(parser, "scheduler", "use one cycle LR scheduler", default=False)
    parser.add_argument("--lr-decay", type=float, default=0.1)
    parser.add_argument("--cycle-up-num-epochs", type=int, default=8)
    parser.add_argument("--cycle-cooldown-num-epochs", type=int, default=4)
    parser.add_argument("--cycle-max-lr", type=float, default=3e-3)
    parser.add_argument("--cycle-final-lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--name",
        type=str,
        default="test",
        help="name or tag for model; will be appended with other info",
    )
    args = parser.parse_args()

    if args.n:
        args.dir_path = "/graphganvol/MPGAN/"

    return args


def init(args):
    os.system(f"mkdir -p {args.dir_path}/coutputs/{args.name}")

    args_dict = vars(args)

    dirs = ["models", "losses", "figs"]

    for dir in dirs:
        args_dict[dir + "_path"] = f"{args.dir_path}/coutputs/{args.name}/{dir}/"
        os.system(f'mkdir -p {args_dict[dir + "_path"]}')

    args_dict["args_path"] = f"{args.dir_path}/coutputs/{args.name}/"
    args_dict["outs_path"] = f"{args.dir_path}/coutputs/{args.name}/"
    args_dict["datasets_path"] = f"{args.dir_path}/datasets/"

    init_logging(args)

    prev_models = [f[:-4] for f in listdir(args.args_path)]  # removing txt part

    if args.name in prev_models:
        logging.info("name already used")
        # if(not args.load_model):
        #     sys.exit()

    if not args.load_model:
        f = open(args.args_path + args.name + ".txt", "w+")
        f.write(str(vars(args)))
        f.close()
    else:
        temp = args.start_epoch, args.num_epochs
        f = open(args.args_path + args.name + ".txt", "r")
        args_dict = vars(args)
        load_args_dict = eval(f.read())
        for key in load_args_dict:
            args_dict[key] = load_args_dict[key]

        args = objectview(args_dict)
        f.close()
        args.load_model = True
        args.start_epoch, args.num_epochs = temp

    args.device = device
    return args


def main(args):
    args = init(args)

    dataset_real = JetNet(
        args.jets,
        data_dir=args.datasets_path,
        num_particles=args.num_hits,
        use_mask=False,
        normalize=False,
        train=False,
    ).data[:50000]
    dataset_fake = torch.tensor(
        np.load(f"{args.dir_path}/trained_models/mp_{args.jets}/gen_jets.npy")
    )

    split = int(50000 * 0.7)

    train_X = torch.cat((dataset_real[:split], dataset_fake[:split]), axis=0)
    test_X = torch.cat((dataset_real[split:], dataset_fake[split:]), axis=0)
    JetNet.normalize_features(train_X, fpnd=True)
    JetNet.normalize_features(test_X, fpnd=True)

    train_Y = torch.cat((torch.ones(split), torch.zeros(split))).long()
    test_Y = torch.cat((torch.ones(50000 - split), torch.zeros(50000 - split))).long()

    train_dataset = TupleDataset(train_X, train_Y)
    test_dataset = TupleDataset(test_X, test_Y)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    C = ParticleNet(args.num_hits, args.node_feat_size, num_classes=2).to(args.device)

    if args.optimizer == "adamw":
        C_optimizer = torch.optim.AdamW(C.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        C_optimizer = torch.optim.Adam(C.parameters(), lr=args.lr)
    elif args.optimizer == "rmsprop":
        C_optimizer = torch.optim.RMSprop(C.parameters(), lr=args.lr)

    if args.scheduler:
        steps_per_epoch = len(train_loader)
        cycle_last_epoch = -1 if not args.load_model else (args.start_epoch * steps_per_epoch) - 1
        cycle_total_epochs = (2 * args.cycle_up_num_epochs) + args.cycle_cooldown_num_epochs

        C_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            C_optimizer,
            max_lr=args.cycle_max_lr,
            pct_start=(args.cycle_up_num_epochs / cycle_total_epochs),
            epochs=cycle_total_epochs,
            steps_per_epoch=steps_per_epoch,
            final_div_factor=args.cycle_final_lr / args.lr,
            anneal_strategy="linear",
            last_epoch=cycle_last_epoch,
        )

    loss = torch.nn.CrossEntropyLoss().to(args.device)

    train_losses = []
    test_losses = []

    def plot_losses(epoch, train_losses, test_losses):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(train_losses)
        ax1.set_title("training")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(test_losses)
        ax2.set_title("testing")

        plt.savefig(args.losses_path + "/" + str(epoch) + ".pdf")
        plt.close()

        try:
            remove(args.losses_path + "/" + str(epoch - 1) + ".pdf")
        except:
            logging.info("couldn't remove loss file")

    def plot_confusion_matrix(
        cm, target_names, dir_path, epoch, title="Confusion matrix", cmap=None, normalize=True
    ):
        """
        given a sklearn confusion matrix (cm), make a nice plot
        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix
        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']
        title:        the text to display at the top of the matrix
        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues
        normalize:    If False, plot the raw numbers
                      If True, plot the proportions
        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph
        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        """

        import itertools

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap("Blues")

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0.0

        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes()
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title + "at epoch" + str(epoch))
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(
                    j,
                    i,
                    "{:0.2f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            else:
                plt.text(
                    j,
                    i,
                    "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True label")
        plt.xlim(-1, len(target_names))
        plt.ylim(-1, len(target_names))
        plt.xlabel("Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass))
        plt.tight_layout()

        plt.savefig(f"{args.figs_path}/{epoch}_cm.pdf")
        plt.close(fig)

        return fig, ax

    def plot_roc(fpr, tpr, epoch, xlim=[0, 0.4], ylim=[1e-6, 1e-2]):
        plt.figure(figsize=(12, 12))
        plt.plot(tpr, fpr)
        # plt.yscale("log")
        plt.xlabel("TPR")
        plt.ylabel("FPR")
        plt.title("ROC")
        # plt.xlim(*xlim)
        # plt.ylim(*ylim)
        plt.savefig(f"{args.figs_path}/{epoch}_roc.pdf")
        # plt.savefig(f"{plotdir}/{name}.pdf", bbox_inches="tight")

    def save_model(epoch):
        torch.save(C.state_dict(), args.models_path + "/C_" + str(epoch) + ".pt")

    def train_C(data, y):
        C.train()
        C_optimizer.zero_grad()

        output = C(data)

        # nll_loss takes class labels as target, so one-hot encoding is not needed
        C_loss = loss(output, y)

        C_loss.backward()
        C_optimizer.step()

        return C_loss.item()

    def test(epoch):
        C.eval()
        test_loss = 0
        correct = 0
        y_outs = []
        logging.info("testing")
        with torch.no_grad():
            for batch_ndx, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
                logging.debug(f"x[0]: {x[0]}, y: {y}")
                output = C(x.to(device))
                y = y.to(device)
                test_loss += loss(output, y).item()
                pred = output.max(1, keepdim=True)[1]
                logging.debug(f"pred: {pred}, output: {output}")

                y_outs.append(output.cpu().numpy())
                correct += pred.eq(y.view_as(pred)).sum()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        y_outs = np.concatenate(y_outs)
        logging.debug(f"y_outs {y_outs}")
        logging.debug(f"y_true {test_dataset[:][1].numpy()}")

        cm = confusion_matrix(test_dataset[:][1].numpy(), y_outs.argmax(axis=1))
        plot_confusion_matrix(cm, ["Real", "MPGAN"], args.losses_path, epoch)

        fpr, tpr, thresholds = roc_curve(test_dataset[:][1].numpy(), softmax(y_outs, axis=1)[:, 0])
        roc_auc = auc(fpr, tpr)

        f = open(f"{args.losses_path}/{epoch}_roc_auc.txt", "w")
        f.write(str(roc_auc))
        f.close()

        f = open(f"{args.losses_path}/{epoch}_tpr.txt", "w")
        f.write(str(tpr))
        f.close()

        f = open(f"{args.losses_path}/{epoch}_fpr.txt", "w")
        f.write(str(fpr))
        f.close()

        plot_roc(fpr, tpr, epoch)

        s = "After {} epochs, on test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            epoch,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
        logging.info(s)

    for i in range(args.start_epoch, args.num_epochs):
        logging.info("Epoch %d %s" % ((i + 1), args.name))
        C_loss = 0
        test(i)
        logging.info("training")
        for batch_ndx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            C_loss += train_C(x.to(device), y.to(device))
            if args.scheduler:
                C_scheduler.step()

        train_losses.append(C_loss / len(train_loader))

        if (i + 1) % 1 == 0:
            save_model(i + 1)
            plot_losses(i + 1, train_losses, test_losses)

    test(args.num_epochs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
