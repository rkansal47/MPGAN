import numpy as np
import matplotlib.pyplot as plt
import utils
import os
from jets_dataset import JetsDataset
import mplhep as hep
# from skhep.math.vectors import LorentzVector
from tqdm import tqdm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import save_outputs
from copy import copy
# plt.switch_backend('macosx')
plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)


dirs = os.listdir('final_models')

num_samples = 50000

samples_dict = {'g': {}, 't': {}, 'q': {}}

for dir in dirs:
    print(dir)
    if dir == '.DS_Store': continue

    model_name = dir.split('_')[0]

    if not (model_name == 'fcpnet' or model_name == 'graphcnnpnet' or model_name == 'mp' or model_name == 'treeganpnet' or model_name == 'pcgan' or model_name == 'mppnet'):
        continue

    samples = np.load('final_models/' + dir + '/samples.npy')[:num_samples]

    path = 'final_models/' + dir + '/'
    files = os.listdir(path)
    for file in files:
        if file[-4:] == ".txt": args_file = file

    args = eval(open(path + args_file).read())
    args['datasets_path'] = 'datasets/'
    args = utils.objectview(args)

    X = JetsDataset(args)

    gen_out_rn, mask_gen = utils.unnorm_data(args, samples, real=False)

    dataset = dir.split('_')[1]

    if model_name == 'fcpnet':
        samples_dict[dataset]['FC'] = (gen_out_rn, mask_gen)
    elif model_name == 'graphcnnpnet':
        samples_dict[dataset]['GraphCNN'] = (gen_out_rn, mask_gen)
    elif model_name == 'treeganpnet':
        samples_dict[dataset]['TreeGAN'] = (gen_out_rn, mask_gen)
    elif model_name == 'pcgan':
        samples_dict[dataset]['PCGAN'] = (gen_out_rn, mask_gen)
    elif model_name == 'mp':
        samples_dict[dataset]['MP'] = (gen_out_rn, mask_gen)
    elif model_name == 'mppnet':
        samples_dict[dataset]['MPPNET'] = (gen_out_rn, mask_gen)



for dataset in samples_dict.keys():
    args = utils.objectview({'datasets_path': 'datasets/', 'ttsplit': 0.7, 'node_feat_size': 3, 'num_hits': 30, 'coords': 'polarrel', 'dataset': 'jets', 'clabels': 0, 'jets': dataset, 'norm': 1, 'mask': True, 'real_only': False, 'model': 'mpgan'})
    X = JetsDataset(args, train=False)
    X = X[:][0]
    X_rn, mask_real = utils.unnorm_data(args, X[:num_samples].cpu().detach().numpy(), real=True)
    samples_dict[dataset]['Real'] = (X_rn, mask_real)

samples_dict['g'].keys()




plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

line_opts = {'Real': {'color': 'red', 'linewidth': 3, 'linestyle': 'solid'},
                'FC': {'color': 'green', 'linewidth': 3, 'linestyle': 'dashdot'},
                'GraphCNN': {'color': 'brown', 'linewidth': 3, 'linestyle': 'dashed'},
                'TreeGAN': {'color': 'orange', 'linewidth': 3, 'linestyle': 'dashdot'},
                'PCGAN': {'color': 'purple', 'linewidth': 3, 'linestyle': (0, (5, 10))},
                'MP': {'color': 'blue', 'linewidth': 3, 'linestyle': 'dashed'},
                # 'MPPNET': {'color': 'purple', 'linewidth': 2, 'linestyle': (0, (5, 10))},
            }

efps = {}
for dataset in samples_dict.keys():
    efps[dataset] = {}
    for key in line_opts.keys():
        samples, mask = samples_dict[dataset][key]
        efps[dataset][key] = utils.efp(utils.objectview({'mask': key == 'Real' or key == 'MP', 'num_hits': 30}), samples, mask, key == 'Real')[:, 0]



%matplotlib inline



fig = plt.figure(figsize=(36, 24))
i = 0
for dataset in samples_dict.keys():
    if dataset == 'g':
        efpbins = np.linspace(0, 0.0013, 51)
        pbins = [np.linspace(-0.3, 0.3, 101), np.linspace(0, 0.1, 101)]
        ylims = [1.3e5, 0.7e5, 4.2e3, 1.75e4]
    elif dataset == 't':
        efpbins = np.linspace(0, 0.0045, 51)
        pbins = [np.arange(-0.5, 0.5, 0.005), np.arange(0, 0.1, 0.001)]
        ylims = [0.35e5, 0.8e5, 3.6e3, 0.35e4]
    else:
        efpbins = np.linspace(0, 0.002, 51)
        pbins = [np.linspace(-0.3, 0.3, 101), np.linspace(0, 0.15, 101)]
        ylims = [2e5, 2.2e5, 6.5e3, 2.5e4]

    mbins = np.linspace(0, 0.225, 51)

    fig.add_subplot(3, 4, i * 4 + 1)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Particle $\eta^{rel}$')
    plt.ylabel('Number of Particles')

    for key in line_opts.keys():
        samples, mask = samples_dict[dataset][key]
        if key == 'MP' or key == 'Real':
            parts = samples[mask]
        else:
            parts = samples.reshape(-1, 3)

        _ = plt.hist(parts[:, 0], pbins[0], histtype='step', label=key, **line_opts[key])

    plt.legend(loc=1, prop={'size': 18}, fancybox=True)
    plt.ylim(0, ylims[0])

    fig.add_subplot(3, 4, i * 4 + 2)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Particle $p_T^{rel}$')
    plt.ylabel('Number of Particles')

    for key in line_opts.keys():
        samples, mask = samples_dict[dataset][key]
        if key == 'MP' or key == 'Real':
            parts = samples[mask]
        else:
            parts = samples.reshape(-1, 3)

        _ = plt.hist(parts[:, 2], pbins[1], histtype='step', label=key, **line_opts[key])

    plt.legend(loc=1, prop={'size': 18}, fancybox=True)
    plt.ylim(0, ylims[1])

    fig.add_subplot(3, 4, i * 4 + 3)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Relative Jet Mass')
    plt.ylabel('Number of Jets')

    for key in line_opts.keys():
        samples, mask = samples_dict[dataset][key]
        masses = utils.jet_features(samples, mask=mask_real)[:, 0]

        _ = plt.hist(masses, mbins, histtype='step', label=key, **line_opts[key])

    plt.legend(loc=1, prop={'size': 18}, fancybox=True)
    plt.ylim(0, ylims[2])


    fig.add_subplot(3, 4, i * 4 + 4)
    plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Jet EFP', x = 0.7)
    plt.ylabel('Number of Jets')

    for key in line_opts.keys():
        _ = plt.hist(efps[dataset][key], efpbins, histtype='step', label=key, **line_opts[key])

    plt.legend(loc=1, prop={'size': 18}, fancybox=True)
    plt.ylim(0, ylims[3])

    i += 1

plt.tight_layout(pad=0.5)
plt.savefig('final_figure_update.pdf', bbox_inches='tight')
plt.show()





def pixelate(jet, mask, im_size, maxR):
    bins = np.linspace(-maxR, maxR, im_size + 1)
    binned_eta = np.digitize(jet[:, 0], bins) - 1
    binned_phi = np.digitize(jet[:, 1], bins) - 1
    pt = jet[:, 2]
    if mask is not None: pt *= mask

    jet_image = np.zeros((im_size, im_size))

    for eta, phi, pt in zip(binned_eta, binned_phi, pt):
        if eta >= 0 and eta < im_size and phi >= 0 and phi < im_size:
            jet_image[phi, eta] += pt

    return jet_image




average_images = {'g': {}, 't': {}, 'q': {}}

from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

order = ['Real', 'MP', 'FC', 'GraphCNN', 'TreeGAN', 'PCGAN']


num_images = 4
np.random.seed(2021)
rand_sample = np.random.randint(num_samples, size=num_images)

im_size = 25
maxR = 0.4
ave_maxR = 0.5

cm = copy(plt.cm.jet)
cm.set_under(color='white')

for dataset in ['g', 't', 'q']:
    fig, axes = plt.subplots(nrows=len(order), ncols=num_images + 1, figsize=(40, 48), gridspec_kw = {'wspace': 0.25, 'hspace':0})

    for j in range(len(order)):
        key = order[j]
        axes[j][0].annotate(key, xy=(0, -1), xytext=(-axes[j][0].yaxis.labelpad - 15, 0),
                    xycoords=axes[j][0].yaxis.label, textcoords='offset points',
                    ha='right', va='center')

        samples, mask = samples_dict[dataset][key]

        rand_samples = samples[rand_sample]
        if mask is not None: rand_mask = mask[rand_sample]

        for i in range(num_images):
            im = axes[j][i].imshow(pixelate(np.flip(rand_samples[i], axis=0), None if mask is None else rand_mask[i], im_size, maxR), cmap=cm, interpolation='nearest', vmin=1e-8, extent=[-maxR, maxR,-maxR, maxR], vmax=0.05)
            axes[j][i].tick_params(which='both', bottom=False, top=False, left=False, right=False)
            axes[j][i].set_xlabel('$\phi^{rel}$')
            axes[j][i].set_ylabel('$\eta^{rel}$')
            # if i == num_images - 1:
            # divider = make_axes_locatable(axes[j][i])
            # cax = divider.append_axes("right", size="5%", pad=0.3)

        # average jet image

        if key not in average_images[dataset]:
            ave_im = np.zeros((im_size, im_size))
            for i in tqdm(range(10000)):
                ave_im += pixelate(samples[i], None if mask is None else mask[i], im_size, ave_maxR)
            ave_im /= 10000
            average_images[dataset][key] = ave_im

        im = axes[j][-1].imshow(average_images[dataset][key], cmap=plt.cm.jet, interpolation='nearest', vmin=1e-8, extent=[-ave_maxR, ave_maxR,-ave_maxR, ave_maxR], vmax=0.05)
        axes[j][-1].set_title('Average Jet Image', pad=5)
        axes[j][-1].tick_params(which='both', bottom=False, top=False, left=False, right=False)
        axes[j][-1].set_xlabel('$\phi^{rel}$')
        axes[j][-1].set_ylabel('$\eta^{rel}$')

        cbar = fig.colorbar(im, ax=axes[j].ravel().tolist(), fraction=0.007)
        cbar.set_label('$p_T^{rel}$')

    fig.tight_layout()
    plt.savefig(f'jet_images_{dataset}.pdf', bbox_inches='tight')
    plt.show()
