import torch
import logging


def mask_manual(args, gen_data, pt_cutoff):
    """applies a zero-mask to particles with pT below ``pt_cutoff``"""

    logging.debug("Before Mask: ")
    logging.debug(gen_data[0])
    if args.mask_real_only:
        mask = torch.ones(gen_data.size(0), gen_data.size(1), 1).to(args.device) - 0.5
    elif args.mask_exp:
        pts = gen_data[:, :, 2].unsqueeze(2)
        upper = (pts > pt_cutoff).float()
        lower = 1 - upper
        exp = torch.exp((pts - pt_cutoff) / abs(pt_cutoff))
        mask = upper + lower * exp - 0.5
    else:
        mask = (gen_data[:, :, 2] > pt_cutoff).unsqueeze(2).float() - 0.5

    gen_data = torch.cat((gen_data, mask), dim=2)
    logging.debug("After Mask: ")
    logging.debug(gen_data[0])
    return gen_data
