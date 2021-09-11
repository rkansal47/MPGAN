# import setGPU

import torch
import setup, utils, save_outputs, evaluation, augment
from jets_dataset import JetsDataset
from torch.utils.data import DataLoader

from tqdm import tqdm

# from parallel import DataParallelModel, DataParallelCriterion

import logging

from guppy import hpy
h = hpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(False)

    args, tqdm_out = setup.init()
    torch.manual_seed(args.seed)
    # args = setup.init()
    args.device = device
    logging.info("Args initalized")

    X = JetsDataset(args, train=True)
    X_loaded = DataLoader(X, shuffle=True, batch_size=args.batch_size, pin_memory=True)

    X_test = JetsDataset(args, train=False)
    X_test_loaded = DataLoader(X_test, batch_size=args.batch_size, pin_memory=True)
    logging.info("Data loaded")

    G, D = setup.models(args)
    if args.model == 'pcgan':
        G_inv, G_pc = setup.pcgan_models(args)
        G_inv.eval()
        G_pc.eval()
        pcgan_train_args = {'sample_points': False}  # no need to sample points while training latent GAN
        pcgan_eval_args = {'sample_points': True, 'G_pc': G_pc}
    else:
        pcgan_train_args = None
        pcgan_eval_args = None

    logging.info("Models loaded")

    G_optimizer, D_optimizer = setup.optimizers(args, G, D)
    logging.info("Optimizers loaded")

    losses = setup.losses(args)

    if args.fpnd: C, mu2, sigma2 = evaluation.load(args, X_test_loaded)

    Y_real = torch.ones(args.batch_size, 1).to(args.device)
    Y_fake = torch.zeros(args.batch_size, 1).to(args.device)

    mse = torch.nn.MSELoss()
    # if args.multi_gpu:
    #     mse = DataParallelCriterion(mse)

    def train_D(data, labels=None, gen_data=None, epoch=0, print_output=False):
        logging.debug("dtrain")
        log = logging.info if print_output else logging.debug

        D.train()
        D_optimizer.zero_grad()
        G.eval()

        run_batch_size = data.shape[0]

        if args.model == 'pcgan': data = G_inv(data.clone())  # run through pre-trained inference network first i.e. find latent representation
        D_real_output = D(data.clone(), labels, epoch=epoch)

        log("D real output: ")
        log(D_real_output[:10])

        if gen_data is None:
            gen_data = utils.gen(args, G, run_batch_size, labels=labels, pcgan_args=pcgan_train_args)

        if args.augment:
            p = args.aug_prob if not args.adaptive_prob else losses['p'][-1]
            data = augment.augment(args, data, p)
            gen_data = augment.augment(args, gen_data, p)

        log("G output: ")
        log(gen_data[:2, :10])

        D_fake_output = D(gen_data, labels, epoch=epoch)

        log("D fake output: ")
        log(D_fake_output[:10])

        D_loss, D_loss_items = utils.calc_D_loss(args, D, data, gen_data, D_real_output, D_fake_output, run_batch_size, Y_real, Y_fake, mse)
        D_loss.backward()

        D_optimizer.step()
        return D_loss_items

    def train_G(data, labels=None, epoch=0):
        logging.debug("gtrain")
        G.train()
        G_optimizer.zero_grad()

        run_batch_size = labels.shape[0] if labels is not None else args.batch_size

        gen_data = utils.gen(args, G, run_batch_size, labels=labels, pcgan_args=pcgan_train_args)

        if args.augment:
            p = args.aug_prob if not args.adaptive_prob else losses['p'][-1]
            gen_data = augment.augment(args, gen_data, p)

        D_fake_output = D(gen_data, labels, epoch=epoch)

        logging.debug("D fake output:")
        logging.debug(D_fake_output[:10])

        G_loss = utils.calc_G_loss(args, D_fake_output, Y_real, run_batch_size, mse)

        G_loss.backward()
        G_optimizer.step()

        return G_loss.item()

    def train():
        logging.info(h.heap())

        if(args.start_epoch == 0 and args.save_zero):
            if args.eval:
                gen_out = evaluation.calc_w1(args, X_test[:][0], G, losses, X_loaded=X_test_loaded, pcgan_args=pcgan_eval_args)
                if args.fpnd: losses['fpnd'].append(evaluation.get_fpnd(args, C, gen_out, mu2, sigma2))
                evaluation.calc_cov_mmd(args, X_test[:][0], gen_out, losses, X_loaded=X_test_loaded)
            else: gen_out = None
            save_outputs.save_sample_outputs(args, D, G, X_test[:args.num_samples][0], 0, losses, X_loaded=X_test_loaded, gen_out=gen_out, pcgan_args=pcgan_eval_args)
            del(gen_out)

        logging.info(h.heap())

        for i in range(args.start_epoch, args.num_epochs):
            logging.info("Epoch {} starting".format(i + 1))
            D_losses = ['Dr', 'Df', 'D']
            if args.gp: D_losses.append('gp')
            epoch_loss = {'G': 0}
            for key in D_losses: epoch_loss[key] = 0

            lenX = len(X_loaded)

            bar = tqdm(enumerate(X_loaded), total=lenX, mininterval=0.1, desc="Epoch {}".format(i + 1))
            for batch_ndx, data in bar:
                if args.clabels or args.mask_c: labels = data[1].to(args.device)
                else: labels = None

                data = data[0].to(args.device)

                if args.num_critic > 1 or (batch_ndx == 0 or (batch_ndx - 1) % args.num_gen == 0):
                    D_loss_items = train_D(data, labels=labels, epoch=i, print_output=(batch_ndx == lenX - 1))  # print outputs for the last iteration of each epoch
                    for key in D_losses: epoch_loss[key] += D_loss_items[key]

                if args.num_critic == 1 or (batch_ndx - 1) % args.num_critic == 0:
                    epoch_loss['G'] += train_G(data, labels=labels, epoch=i)

                if args.bottleneck:
                    if(batch_ndx == 10):
                        return

                if args.break_zero:
                    if(batch_ndx == 0):
                        break

            logging.info("Epoch {} Training Over".format(i + 1))

            for key in D_losses: losses[key].append(epoch_loss[key] / (lenX / args.num_gen))
            losses['G'].append(epoch_loss['G'] / (lenX / args.num_critic))
            for key in epoch_loss.keys(): logging.info("{} loss: {:.3f}".format(key, losses[key][-1]))

            if((i + 1) % args.save_model_epochs == 0):
                optimizers = (D_optimizer, G_optimizer)
                save_outputs.save_models(args, D, G, optimizers, args.name, i + 1)

            if((i + 1) % args.save_epochs == 0):
                if args.eval:
                    gen_out = evaluation.calc_w1(args, X_test[:][0], G, losses, X_loaded=X_test_loaded, pcgan_args=pcgan_eval_args)
                    if args.fpnd: losses['fpnd'].append(evaluation.get_fpnd(args, C, gen_out, mu2, sigma2))
                    evaluation.calc_cov_mmd(args, X_test[:][0], gen_out, losses, X_loaded=X_test_loaded)
                else: gen_out = None
                save_outputs.save_sample_outputs(args, D, G, X_test[:args.num_samples][0], i + 1, losses, X_loaded=X_test_loaded, gen_out=gen_out, pcgan_args=pcgan_eval_args)
                del(gen_out)

            logging.info(h.heap())

    train()


if __name__ == "__main__":
    main()
