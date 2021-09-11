import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm
import logging


class Graph_GAN(nn.Module):
    def __init__(self, gen, args):
        super(Graph_GAN, self).__init__()
        self.args = args

        self.G = gen
        self.D = not gen

        self.args.spectral_norm = self.args.spectral_norm_gen if self.G else self.args.spectral_norm_disc
        self.args.batch_norm = self.args.batch_norm_gen if self.G else self.args.batch_norm_disc
        self.args.mp_iters = self.args.mp_iters_gen if self.G else self.args.mp_iters_disc
        self.args.fe1 = self.args.fe1g if self.G else self.args.fe1d
        if self.G: self.args.dea = False
        if self.D: self.args.lfc = False


        if self.G: self.first_layer_node_size = self.args.latent_node_size if self.args.latent_node_size else self.args.hidden_node_size
        else: self.first_layer_node_size = self.args.node_feat_size

        if not self.args.fe1: self.args.fe1 = self.args.fe.copy()
        self.args.fn1 = self.args.fn.copy()


        # setting up # of nodes in each network layer

        anc = 0     # anc - # of extra params to pass into edge network
        if self.args.pos_diffs:
            if self.args.deltacoords:
                if self.args.coords == 'cartesian':
                    anc += 3
                elif self.args.coords == 'polar' or self.args.coords == 'polarrel':
                    anc += 2
            if self.args.deltar:
                anc += 1

        anc += int(self.args.int_diffs)

        if args.lfc: self.lfc = nn.Linear(self.args.lfc_latent_size, self.args.num_hits * self.first_layer_node_size)

        # edge and node networks
        # both are ModuleLists of ModuleLists
        # with shape: # of MP layers   X   # of FC layers in each MP layer

        self.fe = nn.ModuleList()
        self.fn = nn.ModuleList()

        if(self.args.batch_norm):
            self.bne = nn.ModuleList()
            self.bnn = nn.ModuleList()


        # building first MP layer networks:

        self.args.fe1_in_size = 2 * self.first_layer_node_size + anc + self.args.clabels_first_layer + self.args.mask_fne_np
        self.args.fe1.insert(0, self.args.fe1_in_size)
        self.args.fe1_out_size = self.args.fe1[-1]

        fe_iter = nn.ModuleList()
        if self.args.batch_norm: bne = nn.ModuleList()
        for j in range(len(self.args.fe1) - 1):
            linear = nn.Linear(self.args.fe1[j], self.args.fe1[j + 1])
            fe_iter.append(linear)
            if self.args.batch_norm: bne.append(nn.BatchNorm1d(self.args.fe1[j + 1]))

        self.fe.append(fe_iter)
        if self.args.batch_norm: self.bne.append(bne)


        self.args.fn1.insert(0, self.args.fe1_out_size + self.first_layer_node_size + self.args.clabels_first_layer + self.args.mask_fne_np)
        self.args.fn1.append(self.args.hidden_node_size)

        fn_iter = nn.ModuleList()
        if self.args.batch_norm: bnn = nn.ModuleList()
        for j in range(len(self.args.fn1) - 1):
            linear = nn.Linear(self.args.fn1[j], self.args.fn1[j + 1])
            fn_iter.append(linear)
            if self.args.batch_norm: bnn.append(nn.BatchNorm1d(self.args.fn1[j + 1]))

        self.fn.append(fn_iter)
        if self.args.batch_norm: self.bnn.append(bnn)


        # building networks for the rest of the MP layers:

        self.args.fe_in_size = 2 * self.args.hidden_node_size + anc + self.args.clabels_hidden_layers + self.args.mask_fne_np
        self.args.fe.insert(0, self.args.fe_in_size)
        self.args.fe_out_size = self.args.fe[-1]

        self.args.fn.insert(0, self.args.fe_out_size + self.args.hidden_node_size + self.args.clabels_hidden_layers + self.args.mask_fne_np)
        self.args.fn.append(self.args.hidden_node_size)

        for i in range(self.args.mp_iters - 1):
            fe_iter = nn.ModuleList()
            if self.args.batch_norm: bne = nn.ModuleList()
            for j in range(len(self.args.fe) - 1):
                linear = nn.Linear(self.args.fe[j], self.args.fe[j + 1])
                fe_iter.append(linear)
                if self.args.batch_norm: bne.append(nn.BatchNorm1d(self.args.fe[j + 1]))

            self.fe.append(fe_iter)
            if self.args.batch_norm: self.bne.append(bne)


            fn_iter = nn.ModuleList()
            if self.args.batch_norm: bnn = nn.ModuleList()
            for j in range(len(self.args.fn) - 1):
                linear = nn.Linear(self.args.fn[j], self.args.fn[j + 1])
                fn_iter.append(linear)
                if self.args.batch_norm: bnn.append(nn.BatchNorm1d(self.args.fn[j + 1]))

            self.fn.append(fn_iter)
            if self.args.batch_norm: self.bnn.append(bnn)


        # final disc FCN

        if(self.args.dea):
            self.args.fnd.insert(0, self.args.hidden_node_size + int(self.args.mask_fnd_np))
            self.args.fnd.append(1)

            self.fnd = nn.ModuleList()
            self.bnd = nn.ModuleList()
            for i in range(len(self.args.fnd) - 1):
                linear = nn.Linear(self.args.fnd[i], self.args.fnd[i + 1])
                self.fnd.append(linear)
                if self.args.batch_norm: self.bnd.append(nn.BatchNorm1d(self.args.fnd[i + 1]))


        # initial gen mask FCN

        if self.G and (hasattr(self.args, 'mask_learn') and self.args.mask_learn) or (hasattr(self.args, 'mask_learn_sep') and self.args.mask_learn_sep):
            self.args.fmg.insert(0, self.first_layer_node_size)
            self.args.fmg.append(1 if self.args.mask_learn else self.args.num_hits)

            self.fmg = nn.ModuleList()
            self.bnmg = nn.ModuleList()
            for i in range(len(self.args.fmg) - 1):
                linear = nn.Linear(self.args.fmg[i], self.args.fmg[i + 1])
                self.fmg.append(linear)
                if self.args.batch_norm: self.bnmg.append(nn.BatchNorm1d(self.args.fmg[i + 1]))


        p = self.args.gen_dropout if self.G else self.args.disc_dropout
        self.dropout = nn.Dropout(p=p)

        if self.args.glorot: self.init_params()

        if self.args.spectral_norm:
            for ml in self.fe:
                for i in range(len(ml)):
                    ml[i] = SpectralNorm(ml[i])

            for ml in self.fn:
                for i in range(len(ml)):
                    ml[i] = SpectralNorm(ml[i])

            if self.args.dea:
                for i in range(len(self.fnd)):
                    self.fnd[i] = SpectralNorm(self.fnd[i])

            if self.args.mask_learn:
                for i in range(len(self.fmg)):
                    self.fmg[i] = SpectralNorm(self.fmg[i])

        if(self.args.lfc):
            logging.info("lfcn: ")
            logging.info(self.lfc)

        logging.info("fe: ")
        logging.info(self.fe)

        logging.info("fn: ")
        logging.info(self.fn)

        if(self.args.dea):
            logging.info("fnd: ")
            logging.info(self.fnd)

        if self.G and (hasattr(self.args, 'mask_learn') and self.args.mask_learn) or (hasattr(self.args, 'mask_learn_sep') and self.args.mask_learn_sep):
            logging.info("fmg: ")
            logging.info(self.fmg)


    def forward(self, x, labels=None, epoch=0):
        batch_size = x.shape[0]

        logging.debug(f"x: {x}")

        if self.args.lfc:
            x = self.lfc(x).reshape(batch_size, self.args.num_hits, self.first_layer_node_size)
            logging.debug(f"LFC'd x: {x}")

        try:
            mask_bool = (self.D and (self.args.mask_manual or self.args.mask_real_only or self.args.mask_learn or self.args.mask_c or self.args.mask_learn_sep)) \
                        or (self.G and (self.args.mask_learn or self.args.mask_c or self.args.mask_learn_sep)) \
                        and epoch >= self.args.mask_epoch

            if self.D and (mask_bool or self.args.mask_fnd_np): mask = x[:, :, 3:4] + 0.5
            if self.D and (self.args.mask_manual or self.args.mask_learn or self.args.mask_c or self.args.mask_learn_sep): x = x[:, :, :3]

            if self.G and self.args.mask_learn:
                mask = F.leaky_relu(self.fmg[0](x), negative_slope=self.args.leaky_relu_alpha)
                if(self.args.batch_norm): mask = self.bnmg[0](mask)
                mask = self.dropout(mask)
                for i in range(len(self.fmg) - 1):
                    mask = F.leaky_relu(self.fmg[i + 1](mask), negative_slope=self.args.leaky_relu_alpha)
                    if(self.args.batch_norm): mask = self.bnmg[i](mask)
                    mask = self.dropout(mask)

                mask = torch.sign(mask) if self.args.mask_learn_bin else torch.sigmoid(mask)
                logging.debug("gen mask \n {}".format(mask[:2, :, 0]))

            if self.G and self.args.mask_c:
                nump = (labels[:, self.args.clabels] * self.args.num_hits).int() - 1
                mask = (x[:, :, 0].argsort(1).argsort(1) <= nump.unsqueeze(1)).unsqueeze(2).float()
                logging.debug("x \n {} \n num particles \n {} \n gen mask \n {}".format(x[:2, :, 0], nump[:2], mask[:2, :, 0]))

            if self.args.mask_fne_np:
                nump = torch.mean(mask, dim=1)
                logging.debug("nump \n {}".format(nump[:2]))

            if self.G and self.args.mask_learn_sep:
                nump = x[:, -1, :]
                x = x[:, :-1, :]

                for i in range(len(self.fmg)):
                    nump = F.leaky_relu(self.fmg[i](nump), negative_slope=self.args.leaky_relu_alpha)
                    if(self.args.batch_norm): nump = self.bnmg[i](nump)
                    nump = self.dropout(nump)

                nump = torch.argmax(nump, dim=1)
                mask = (x[:, :, 0].argsort(1).argsort(1) <= nump.unsqueeze(1)).unsqueeze(2).float()

                logging.debug("x \n {} \n num particles \n {} \n gen mask \n {}".format(x[:2, :, 0], nump[:2], mask[:2, :, 0]))

        except AttributeError:
            mask_bool = False

        if not mask_bool: mask = None

        for i in range(self.args.mp_iters):
            clabel_iter = self.args.clabels and ((i == 0 and self.args.clabels_first_layer) or (i and self.args.clabels_hidden_layers))

            node_size = x.size(2)
            fe_in_size = self.args.fe_in_size if i else self.args.fe1_in_size
            fe_out_size = self.args.fe_out_size if i else self.args.fe1_out_size

            if clabel_iter: fe_in_size -= self.args.clabels
            if self.args.mask_fne_np: fe_in_size -= 1

            # message passing
            A, A_mask = self.getA(x, i, batch_size, fe_in_size, mask_bool, mask)

            # logging.debug('A \n {} \n A_mask \n {}'.format(A[:2, :10], A_mask[:2, :10]))

            num_knn = self.args.num_hits if (hasattr(self.args, 'fully_connected') and self.args.fully_connected) else self.args.num_knn

            if (A != A).any(): logging.warning("Nan values in A \n x: \n {} \n A: \n {}".format(x, A))

            # NEED TO FIX FOR MASK-FNE-NP + CLABELS (probably just labels --> labels[:, :self.args.clabels])
            if clabel_iter: A = torch.cat((A, labels.repeat(self.args.num_hits * num_knn, 1)), axis=1)
            if self.args.mask_fne_np: A = torch.cat((A, nump.repeat(self.args.num_hits * num_knn, 1)), axis=1)

            for j in range(len(self.fe[i])):
                A = F.leaky_relu(self.fe[i][j](A), negative_slope=self.args.leaky_relu_alpha)
                if(self.args.batch_norm): A = self.bne[i][j](A)  # try before activation
                A = self.dropout(A)

            if (A != A).any(): logging.warning("Nan values in A after message passing \n x: \n {} \n A: \n {}".format(x, A))

            # message aggregation into new features
            A = A.view(batch_size, self.args.num_hits, num_knn, fe_out_size)
            if mask_bool:
                if self.args.fully_connected: A = A * mask.unsqueeze(1)
                else: A = A * A_mask.reshape(batch_size, self.args.num_hits, num_knn, 1)

            # logging.debug('A \n {}'.format(A[:2, :10]))

            A = torch.sum(A, 2) if self.args.sum else torch.mean(A, 2)
            x = torch.cat((A, x), 2).view(batch_size * self.args.num_hits, fe_out_size + node_size)

            if (x != x).any(): logging.warning("Nan values in x after message passing \n x: \n {} \n A: \n {}".format(x, A))

            if clabel_iter: x = torch.cat((x, labels.repeat(self.args.num_hits, 1)), axis=1)
            if self.args.mask_fne_np: x = torch.cat((x, nump.repeat(self.args.num_hits, 1)), axis=1)

            for j in range(len(self.fn[i]) - 1):
                x = F.leaky_relu(self.fn[i][j](x), negative_slope=self.args.leaky_relu_alpha)
                if(self.args.batch_norm): x = self.bnn[i][j](x)
                x = self.dropout(x)

            x = self.dropout(self.fn[i][-1](x))
            x = x.view(batch_size, self.args.num_hits, self.args.hidden_node_size)

            if (x != x).any(): logging.warning("Nan values in x after fn \n x: \n {} \n A: \n {}".format(x, A))

        if(self.G):
            x = torch.tanh(x[:, :, :self.args.node_feat_size]) if self.args.gtanh else x[:, :, :self.args.node_feat_size]
            if mask_bool:
                x = torch.cat((x, mask - 0.5), dim=2)
            if hasattr(self.args, 'mask_feat_bin') and self.args.mask_feat_bin:
                mask = (x[:, :, 3:4] < 0).float() - 0.5     # inversing mask sign for positive mask initializations
                x = torch.cat((x[:, :, :3], mask), dim=2)
            return x
        else:
            if(self.args.dea):
                if mask_bool:
                    x = x * mask
                    x = torch.sum(x, 1)
                    if not self.args.sum: x = x / (torch.sum(mask, 1) + 1e-12)
                else: x = torch.sum(x, 1) if self.args.sum else torch.mean(x, 1)

                if hasattr(self.args, 'mask_fnd_np') and self.args.mask_fnd_np:
                    num_particles = torch.mean(mask, dim=1)
                    x = torch.cat((num_particles, x), dim=1)

                for i in range(len(self.fnd) - 1):
                    x = F.leaky_relu(self.fnd[i](x), negative_slope=self.args.leaky_relu_alpha)
                    if(self.args.batch_norm): x = self.bnd[i](x)
                    x = self.dropout(x)

                x = self.dropout(self.fnd[-1](x))
            else:
                x = x[:, :, :1]
                if mask_bool:
                    logging.debug("D output pre mask")
                    logging.debug(mask[:2, :, 0])
                    logging.debug(x[:2, :, 0])
                    x = x * mask
                    logging.debug("post mask")
                    logging.debug(x[:2, :, 0])
                    x = torch.sum(x, 1) / (torch.sum(mask, 1) + 1e-12)
                else:
                    x = torch.mean(x, 1)

            return x if (self.args.loss == 'w' or self.args.loss == 'hinge') else torch.sigmoid(x)

    def getA(self, x, i, batch_size, fe_in_size, mask_bool, mask):
        node_size = x.size(2)
        num_coords = 3 if self.args.coords == 'cartesian' else 2

        A_mask = None

        if self.args.fully_connected:
            x1 = x.repeat(1, 1, self.args.num_hits).view(batch_size, self.args.num_hits * self.args.num_hits, node_size)
            x2 = x.repeat(1, self.args.num_hits, 1)

            if self.args.pos_diffs:
                if self.args.all_ef and not (self.D and i == 0): diffs = x2 - x1  # for first iteration of D message passing use only physical coords
                else: diffs = x2[:, :, :num_coords] - x1[:, :, :num_coords]
                dists = torch.norm(diffs + 1e-12, dim=2).unsqueeze(2)

                if self.args.deltar and self.args.deltacoords:
                    A = torch.cat((x1, x2, diffs, dists), 2)
                elif self.args.deltar:
                    A = torch.cat((x1, x2, dists), 2)
                elif self.args.deltacoords:
                    A = torch.cat((x1, x2, diffs), 2)

                A = A.view(batch_size * self.args.num_hits * self.args.num_hits, fe_in_size)
            else:
                A = torch.cat((x1, x2), 2).view(batch_size * self.args.num_hits * self.args.num_hits, fe_in_size)

        else:
            x1 = x.repeat(1, 1, self.args.num_hits).view(batch_size, self.args.num_hits * self.args.num_hits, node_size)

            if mask_bool:
                mul = 1e4  # multiply masked particles by this so they are not selected as a nearest neighbour
                x2 = (((1 - mul) * mask + mul) * x).repeat(1, self.args.num_hits, 1)
            else:
                x2 = x.repeat(1, self.args.num_hits, 1)

            if (self.args.all_ef or not self.args.pos_diffs) and not (self.D and i == 0): diffs = x2 - x1  # for first iteration of D message passing use only physical coords
            else: diffs = x2[:, :, :num_coords] - x1[:, :, :num_coords]

            dists = torch.norm(diffs + 1e-12, dim=2).reshape(batch_size, self.args.num_hits, self.args.num_hits)

            sorted = torch.sort(dists, dim=2)
            self_loops = int(self.args.self_loops is False)

            # logging.debug("x \n {} \n x1 \n {} \n x2 \n {} \n diffs \n {} \n dists \n {} \n sorted[0] \n {} \n sorted[1] \n {}".format(x[0], x1[0], x2[0], diffs[0], dists[0], sorted[0][0], sorted[0][1]))

            dists = sorted[0][:, :, self_loops:self.args.num_knn + self_loops].reshape(batch_size, self.args.num_hits * self.args.num_knn, 1)
            sorted = sorted[1][:, :, self_loops:self.args.num_knn + self_loops].reshape(batch_size, self.args.num_hits * self.args.num_knn, 1)

            sorted.reshape(batch_size, self.args.num_hits * self.args.num_knn, 1).repeat(1, 1, node_size)

            x1_knn = x.repeat(1, 1, self.args.num_knn).view(batch_size, self.args.num_hits * self.args.num_knn, node_size)

            if mask_bool:
                x2_knn = torch.gather(torch.cat((x, mask), dim=2), 1, sorted.repeat(1, 1, node_size + 1))
                A_mask = x2_knn[:, :, -1:]
                x2_knn = x2_knn[:, :, :-1]
            else:
                x2_knn = torch.gather(x, 1, sorted.repeat(1, 1, node_size))

            if self.args.pos_diffs:
                A = torch.cat((x1_knn, x2_knn, dists), dim=2)
            else:
                A = torch.cat((x1_knn, x2_knn), dim=2)
            # logging.debug("A \n {} \n".format(A[0]))

        return A, A_mask

    def init_params(self):
        logging.info("glorot-ing")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight, self.args.glorot)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
