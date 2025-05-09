# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import cv2
import pyspng
import glob
import os
import re
import random
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import legacy
from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dpath', help='the path of the input image', required=True)
@click.option('--mpath', help='the path of the mask')
@click.option('--resolution', type=int, help='resolution of input image', default=512, show_default=True)
# @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
# @click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--batch', is_flag=True, help='Process all images in the directory')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    dpath: str,
    mpath: Optional[str],
    resolution: int,
    # truncation_psi: float,
    # noise_mode: str,
    outdir: str,
    batch: bool,
):
    # """
    # Generate images using pretrained network pickle.
    # """
    # seed = 240  # pick up a random number
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    # print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=resolution, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device)

    if batch:
        img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg'))
        mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg')) if mpath else None
        assert mask_list is None or len(img_list) == len(mask_list), 'Image and mask count mismatch'
        for i, ipath in enumerate(img_list):
            iname = os.path.basename(ipath)
            print(f'[DEBUG] Processing: {iname}')
            image = read_image(ipath)
            image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)
            if mask_list:
                mask = cv2.imread(mask_list[i], cv2.IMREAD_GRAYSCALE)
                mask = (mask > 128).astype(np.uint8)  # [0, 1]
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
            else:
                mask = RandomMask(resolution).astype(np.uint8)
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)
            z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            with torch.no_grad():
                output = G(image, mask, z, label, truncation_psi=1, noise_mode='const')
                output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                output = output[0].cpu().numpy()
                outpath = f'{outdir}/{iname}'
                print(f'[DEBUG] Saving image to {outpath}')
                PIL.Image.fromarray(output, 'RGB').save(outpath)

def read_image(image_path, resize_to=None):
    with open(image_path, 'rb') as f:
        if pyspng is not None and image_path.endswith('.png'):
            image = pyspng.load(f.read())
        else:
            image = PIL.Image.open(f)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, axis=2)
    if resize_to:
        image = PIL.Image.fromarray(image).resize((resize_to, resize_to), PIL.Image.LANCZOS)
        image = np.array(image)
    image = image.transpose(2, 0, 1)
    image = image[:3]
    return image


if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
