import numpy as np
import torch
import utils
from jets_dataset import JetsDataset
from model import Graph_GAN
from ext_models import rGANG, GraphCNNGANG, TreeGANG
from pcgan_model import latent_G
from pcgan_model import G as G_pc


import os
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# models = [
#     ['mplfc_g', 1410, '219_g30_mask_c_lfc_dea_no_pos_diffs/'],
#     ['mplfc_t', 1920, '207_t30_mask_c_lrx2_lfc_dea_no_pos_diffs/'],
#     ['mplfc_q', 2475, '231_q30_mask_c_lrx05_lfc_dea_no_pos_diffs/'],
#     ['fcmp_g', 330, '248_g30_dea_no_pos_diffs_rgang_mpgand/'],
#     ['fcmp_t', 185, '249_t30_lrx2_dea_no_pos_diffs_rgang_mpgand/'],
#     ['fcmp_q', 2205, '250_q30_lrx05_dea_no_pos_diffs_rgang_mpgand/'],
#     ['graphcnnmp_g', 1215, '236_g30_dea_no_pos_diffs_graphcnngang_mpgand/'],
#     ['graphcnnmp_t', 210, '237_t30_lrx2_dea_no_pos_diffs_graphcnngang_mpgand/'],
#     ['graphcnnmp_q', 315, '238_q30_lrx05_dea_no_pos_diffs_graphcnngang_mpgand/'],
#     ['mpfc_g', 950, '239_g30_lfc_dea_no_pos_diffs_mpgang_rgand/'],
#     ['mpfc_t', 50, '240_t30_lrx2_dea_no_pos_diffs_mpgang_rgand/'],
#     ['mpfc_q', 765, '241_q30_lrx05_dea_no_pos_diffs_mpgang_rgand/'],
# ]
#
# model_list = os.listdir('models')
# for model in models:
#     model[2] = model[2][:-1]
#     if not model[2] in model_list: continue
#     os.makedirs('./final_models/' + model[0])
#     shutil.copy(f"models/{model[2]}/G_{model[1]}.pt", f'./final_models/{model[0]}/')
#     shutil.copy(f"models/{model[2]}/D_{model[1]}.pt", f'./final_models/{model[0]}/')
#     shutil.copy(f"args/{model[2]}.txt", f'./final_models/{model[0]}/')
#
#     args = eval(open(f"args/{model[2]}.txt").read())
#     args['device'] = device
#     args['batch_size'] = 1024
#     args['datasets_path'] = 'datasets/'
#     args = utils.objectview(args)
#
# if model[0][:2] == 'fc':
#     G = rGANG(args).to(device)
# elif model[0][:2] == 'gr':
#     G = GraphCNNGANG(args).to(device)
# elif model[0][:2] == 'mp':
#     G = Graph_GAN(True, args).to(device)
#
#     X = JetsDataset(args)
#
#     labels = X[:][1].to(device)
#
#     G.load_state_dict(torch.load(f"models/{model[2]}/G_{model[1]}.pt", map_location=device))
#
#     G.eval()
#     gen_out = utils.gen_multi_batch(args, G, 10, labels=labels, use_tqdm=True)
#     np.save(f"./final_models/{model[0]}/samples.npy", gen_out)


dirs = os.listdir('final_models')
# if '.DS_Store' in dirs: del dirs[0]

for dir in dirs:
    print(dir)
    if dir == '.DS_Store': continue

    path = 'final_models/' + dir + '/'
    files = os.listdir(path)
    for file in files:
        if file[-4:] == ".txt": args_file = file
        if file[0] == 'G' and file[-3:] == '.pt': G_file = file

    args = eval(open(path + args_file).read())
    args['device'] = device
    args['batch_size'] = 1024
    args['datasets_path'] = 'datasets/'
    args = utils.objectview(args)


    model_name = dir.split('_')[0]
    print(model_name)

    if 'treegan' in model_name:
        continue
        G = TreeGANG(args.treegang_features, args.treegang_degrees, args.treegang_support).to(device)
        pcgan_args = None
    elif model_name == 'pcgan':
        Gpc = G_pc(args.node_feat_size, args.pcgan_z1_dim, args.pcgan_z2_dim).to(device)
        Gpc.load_state_dict(torch.load(f"models/pcgan/pcgan_G_pc_{args.jets}.pt", map_location=args.device))
        G = latent_G(args.pcgan_latent_dim, args.pcgan_z1_dim).to(device)
        pcgan_args = {'sample_points': True, 'G_pc': Gpc}
    else: continue

    # if model_name == 'fcpnet':
    #     G = rGANG(args).to(device)
    # elif model_name == 'graphcnnpnet':
    #     G = GraphCNNGANG(args).to(device)
    # elif model_name == 'mppnet':
    #     G = Graph_GAN(True, args).to(device)
    # else: continue

    # X = JetsDataset(args)

    # labels = X[:][1].to(device)
    labels = None

    print(G)
    print(path + G_file)
    G.load_state_dict(torch.load(path + G_file, map_location=device))

    G.eval()
    gen_out = utils.gen_multi_batch(args, G, 100000, labels=labels, use_tqdm=True, pcgan_args=pcgan_args)
    np.save(path + "samples.npy", gen_out)
