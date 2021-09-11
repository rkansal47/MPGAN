import torch

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import numpy as np
from scipy import linalg

import logging
import tqdm
import io

from torch.distributions.normal import Normal

import energyflow as ef

from sys import platform
import awkward as ak

from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)


# from https://stackoverflow.com/questions/14897756/python-progress-bar-through-logging-module/38895482#38895482
class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


# from https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    reset = "\x1b[0m"
    info_format = '%(asctime)s %(message)s'
    debug_format = '%(asctime)s [%(filename)s:%(lineno)d in %(funcName)s] %(message)s'

    def __init__(self, args):
        if args.log_file == 'stdout':
            self.FORMATS = {
                logging.DEBUG: self.blue + self.debug_format + self.reset,
                logging.INFO: self.grey + self.info_format + self.reset,
                logging.WARNING: self.yellow + self.debug_format + self.reset,
                logging.ERROR: self.red + self.debug_format + self.reset,
                logging.CRITICAL: self.bold_red + self.debug_format + self.reset
            }
        else:
            self.FORMATS = {
                logging.DEBUG: self.debug_format,
                logging.INFO: self.info_format,
                logging.WARNING: self.debug_format,
                logging.ERROR: self.debug_format,
                logging.CRITICAL: self.debug_format
            }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%d/%m %H:%M:%S')
        return formatter.format(record)


def add_bool_arg(parser, name, help, default=False, no_name=None):
    varname = '_'.join(name.split('-'))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=varname, action='store_true', help=help)
    if(no_name is None):
        no_name = 'no-' + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument('--' + no_name, dest=varname, action='store_false', help=no_help)
    parser.set_defaults(**{varname: default})


def mask_manual(args, gen_data):
    logging.debug("Before Mask: ")
    logging.debug(gen_data[0])
    if args.mask_real_only:
        mask = torch.ones(gen_data.size(0), gen_data.size(1), 1).to(args.device) - 0.5
    elif args.mask_exp:
        pts = gen_data[:, :, 2].unsqueeze(2)
        upper = (pts > args.pt_cutoff).float()
        lower = 1 - upper
        exp = torch.exp((pts - args.pt_cutoff) / abs(args.pt_cutoff))
        mask = upper + lower * exp - 0.5
    else:
        mask = (gen_data[:, :, 2] > args.pt_cutoff).unsqueeze(2).float() - 0.5

    gen_data = torch.cat((gen_data, mask), dim=2)
    logging.debug("After Mask: ")
    logging.debug(gen_data[0])
    return gen_data


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def gen(args, G, num_samples=0, noise=None, labels=None, X_loaded=None, pcgan_args=None):
    dist = Normal(torch.tensor(0.).to(args.device), torch.tensor(args.sd).to(args.device))

    if(noise is None):
        if args.model == 'mpgan':
            if args.lfc:
                noise = dist.sample((num_samples, args.lfc_latent_size))
            else:
                extra_noise_p = int(hasattr(args, 'mask_learn_sep') and args.mask_learn_sep)
                noise = dist.sample((num_samples, args.num_hits + extra_noise_p, args.latent_node_size if args.latent_node_size else args.hidden_node_size))
        elif args.model == 'rgan' or args.model == 'graphcnngan':
            noise = dist.sample((num_samples, args.latent_dim))
        elif args.model == 'treegan':
            noise = [dist.sample((num_samples, 1, args.treegang_features[0]))]
        elif args.model == 'pcgan':
            noise = dist.sample((num_samples, args.pcgan_latent_dim))
            if pcgan_args['sample_points']:
                point_noise = Normal(torch.tensor(0.).to(args.device), torch.tensor(1.).to(args.device)).sample([num_samples, args.num_hits, args.pcgan_z2_dim])

    else: num_samples = noise.size(0)

    if (args.clabels or args.mask_c) and labels is None:
        labels = next(iter(X_loaded))[1].to(args.device)
        while(labels.size(0) < num_samples):
            labels = torch.cat((labels, next(iter(X_loaded))[1]), axis=0)
        labels = labels[:num_samples]

    gen_data = G(noise, labels)
    if args.mask_manual: gen_data = mask_manual(args, gen_data)
    if args.model == 'pcgan' and pcgan_args['sample_points']:
        gen_data = pcgan_args['G_pc'](gen_data.unsqueeze(1), point_noise)

    logging.debug(gen_data[0, :10])
    return gen_data


def gen_multi_batch(args, G, num_samples, noise=None, labels=None, X_loaded=None, use_tqdm=False, pcgan_args=None):
    gen_out = gen(args, G, num_samples=args.batch_size, labels=None if labels is None else labels[:args.batch_size], X_loaded=X_loaded, pcgan_args=pcgan_args).cpu().detach().numpy()
    if use_tqdm:
        for i in tqdm.tqdm(range(int(num_samples / args.batch_size))):
            gen_out = np.concatenate((gen_out, gen(args, G, num_samples=args.batch_size, labels=None if labels is None else labels[args.batch_size * (i + 1):args.batch_size * (i + 2)], X_loaded=X_loaded, pcgan_args=pcgan_args).cpu().detach().numpy()), 0)
    else:
        for i in range(int(num_samples / args.batch_size)):
            gen_out = np.concatenate((gen_out, gen(args, G, num_samples=args.batch_size, labels=None if labels is None else labels[args.batch_size * (i + 1):args.batch_size * (i + 2)], X_loaded=X_loaded, pcgan_args=pcgan_args).cpu().detach().numpy()), 0)
    gen_out = gen_out[:num_samples]

    return gen_out


# from https://github.com/EmilienDupont/wgan-gp
def gradient_penalty(args, D, real_data, generated_data, batch_size):
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1).to(args.device) if not args.model == 'pcgan' else torch.rand(batch_size, 1).to(args.device)
    alpha = alpha.expand_as(real_data)
    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = Variable(interpolated, requires_grad=True).to(args.device)

    del alpha
    torch.cuda.empty_cache()

    # Calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).to(args.device), create_graph=True, retain_graph=True, allow_unused=True)[0].to(args.device)
    gradients = gradients.contiguous()

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    gp = args.gp * ((gradients_norm - 1) ** 2).mean()
    return gp


bce = torch.nn.BCELoss()
# mse = torch.nn.MSELoss()


def calc_D_loss(args, D, data, gen_data, real_outputs, fake_outputs, run_batch_size, Y_real, Y_fake, mse):
    if(args.loss == 'og' or args.loss == 'ls'):
        if args.label_smoothing:
            Y_real = torch.empty(run_batch_size).uniform_(0.7, 1.2).to(args.device)
            Y_fake = torch.empty(run_batch_size).uniform_(0.0, 0.3).to(args.device)
        else:
            Y_real = Y_real[:run_batch_size]
            Y_fake = Y_fake[:run_batch_size]

        # randomly flipping labels for D
        if args.label_noise:
            Y_real[torch.rand(run_batch_size) < args.label_noise] = 0
            Y_fake[torch.rand(run_batch_size) < args.label_noise] = 1

    if(args.loss == 'og'):
        D_real_loss = bce(real_outputs, Y_real)
        D_fake_loss = bce(fake_outputs, Y_fake)
    elif(args.loss == 'ls'):
        D_real_loss = mse(real_outputs, Y_real)
        D_fake_loss = mse(fake_outputs, Y_fake)
    elif(args.loss == 'w'):
        D_real_loss = -real_outputs.mean()
        D_fake_loss = fake_outputs.mean()
    elif(args.loss == 'hinge'):
        D_real_loss = torch.nn.ReLU()(1.0 - real_outputs).mean()
        D_fake_loss = torch.nn.ReLU()(1.0 + fake_outputs).mean()

    D_loss = D_real_loss + D_fake_loss

    if(args.gp):
        gp = gradient_penalty(args, D, data, gen_data, run_batch_size)
        gpitem = gp.item()
        D_loss += gp
    else: gpitem = None

    return (D_loss, {'Dr': D_real_loss.item(), 'Df': D_fake_loss.item(), 'gp': gpitem, 'D': D_real_loss.item() + D_fake_loss.item()})


def calc_G_loss(args, fake_outputs, Y_real, run_batch_size, mse):
    if(args.loss == 'og'):
        G_loss = bce(fake_outputs, Y_real[:run_batch_size])
    elif(args.loss == 'ls'):
        G_loss = mse(fake_outputs, Y_real[:run_batch_size])
    elif(args.loss == 'w' or args.loss == 'hinge'):
        G_loss = -fake_outputs.mean()

    return G_loss


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

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        logging.debug(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def rand_mix(args, X1, X2, p):
    if p == 1: return X1

    assert X1.size(0) == X2.size(0), "Error: different batch sizes of rand mix data"
    batch_size = X1.size(0)

    rand = torch.rand(batch_size, 1, 1).to(args.device)
    mix = torch.zeros(batch_size, 1, 1).to(args.device)
    mix[rand < p] = 1

    return X1 * (1 - mix) + X2 * mix


def jet_features(jets, mask_bool=False, mask=None):
    vecs = ak.zip({
            "pt": jets[:, :, 2:3],
            "eta": jets[:, :, 0:1],
            "phi": jets[:, :, 1:2],
            "mass": ak.full_like(jets[:, :, 2:3], 0),
            }, with_name="PtEtaPhiMLorentzVector")

    sum_vecs = vecs.sum(axis=1)

    jf = np.nan_to_num(np.array(np.concatenate((sum_vecs.mass, sum_vecs.pt), axis=1)))

    return jf


def unnorm_data(args, jets, real=True, rem_zero=True):
    if args.mask:
        if real: mask = (jets[:, :, 3] + 0.5) >= 1
        else: mask = (jets[:, :, 3] + 0.5) >= 0.5
    else:
        mask = None

    if args.coords == 'cartesian':
        jets = jets * args.maxp / args.norm
    else:
        jets = jets[:, :, :3]
        jets = jets / args.norm
        jets[:, :, 2] += 0.5
        jets *= args.maxepp[:3]

    if not real and rem_zero:
        for i in range(len(jets)):
            for j in range(args.num_hits):
                if jets[i][j][2] < 0:
                    jets[i][j][2] = 0
                if mask is not None and not mask[i][j]:
                    for k in range(3):
                        jets[i][j][k] = 0

    return jets, mask


# convert our format (eta, phi, pt) into that of the energyflow library (pt, y, phi)
def ef_format(jets):
    pt = np.expand_dims(jets[:, :, 2], 2)
    eta = np.expand_dims(jets[:, :, 0], 2)
    phi = np.expand_dims(jets[:, :, 1], 2)
    # for mass = 0, y = eta
    return np.concatenate((pt, eta, phi, np.zeros((jets.shape[0], jets.shape[1], 1))), axis=2)


def efp(args, jets, mask=None, real=True):
    efpset = ef.EFPSet(('n==', 4), ('d==', 4), ('p==', 1), measure='hadr', beta=1, normed=None, coords='ptyphim')

    efp_format = ef_format(jets)

    if not real and args.mask:
        for i in range(jets.shape[0]):
            for j in range(args.num_hits):
                if not mask[i][j]:
                    for k in range(4):
                        efp_format[i][j][k] = 0

    logging.info("Batch Computing")

    return efpset.batch_compute(efp_format)
