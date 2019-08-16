'''
Useful helper functions
'''

import os
from os.path import join as fullfile
import numpy as np
import cv2 as cv
import math
import random
import skimage.util
import torch
import torch.nn as nn
import pytorch_ssim
from torch.utils.data import DataLoader
from CompenNetPlusplusDataset import SimpleDataset


# set random number generators' seeds
def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# read images using multi-thread
def readImgsMT(img_dir, size=None, index=None):
    img_dataset = SimpleDataset(img_dir, index=index, size=size)
    data_loader = DataLoader(img_dataset, batch_size=len(img_dataset), shuffle=False, drop_last=False, num_workers=4)

    for i, imgs in enumerate(data_loader):
        # imgs.permute((0, 3, 1, 2)).to('cpu', dtype=torch.float32)/255
        # convert to torch.Tensor
        return imgs.permute((0, 3, 1, 2)).float().div(255)


# create an image montage from a (row, col, C, N) np.ndarray or (N, row, col, C) tensor
def montage(im_in, grid_shape=None, padding_width=5, fill=(1, 1, 1), multichannel=True):
    if type(im_in) is np.ndarray:
        assert im_in.ndim == 4, 'requires a 4-D array with shape (row, col, C, N)'
        im = im_in.transpose(3, 0, 1, 2)

    elif type(im_in) is torch.Tensor:
        assert im_in.ndimension() == 4, 'requires a 4-D tensor with shape (N, C, row, col)'

        if im_in.device.type == 'cuda':
            im_in = im_in.cpu()
        if im_in.requires_grad:
            im_in = im_in.detach()
        im = im_in.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)

    if grid_shape is None:
        num_rows = math.ceil(math.sqrt(im.shape[0]))
        num_cols = math.ceil(im.shape[0] / num_rows)
        grid_shape = (num_rows, num_cols)
    else:
        num_rows = grid_shape[0]
        num_cols = grid_shape[1]
        if num_rows == -1:
            grid_shape = (im.shape[0] / num_cols, num_cols)
        elif num_cols == -1:
            grid_shape = (num_rows, im.shape[0] / num_rows)

    im_out = skimage.util.montage(im, rescale_intensity=False, multichannel=multichannel, padding_width=padding_width, fill=fill,
                                  grid_shape=grid_shape)

    return im_out


# Same as np.repeat, while torch.repeat works as np.tile
def repeat_np(a, repeats, dim):
    '''
    Substitute for numpy's repeat function. Source from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2
    torch.repeat([1,2,3], 2) --> [1, 2, 3, 1, 2, 3]
    np.repeat([1,2,3], repeats=2, axis=0) --> [1, 1, 2, 2, 3, 3]

    :param a: tensor
    :param repeats: number of repeats
    :param dim: dimension where to repeat
    :return: tensor with repitions
    '''

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = repeats
    a = a.repeat(*(repeat_idx))
    if a.is_cuda:  # use cuda-device if input was on cuda device already
        order_index = torch.cuda.LongTensor(
            torch.cat([init_dim * torch.arange(repeats, device=a.device) + i for i in range(init_dim)]))
    else:
        order_index = torch.LongTensor(
            torch.cat([init_dim * torch.arange(repeats) + i for i in range(init_dim)]))

    return torch.index_select(a, dim, order_index)


# save 4D np.ndarray or torch tensor to image files
def saveImgs(inputData, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    if type(inputData) is torch.Tensor:
        if inputData.device.type == 'cuda':
            imgs = inputData.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        else:
            imgs = inputData.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)

    else:
        imgs = inputData

    # imgs must have a shape of (N, row, col, C)
    imgs = np.uint8(imgs[:, :, :, ::-1] * 255)  # convert to BGR and uint8 for opencv
    for i in range(imgs.shape[0]):
        file_name = 'img_{:04d}.png'.format(i + 1)
        cv.imwrite(fullfile(dir, file_name), imgs[i, :, :, :])  # faster than PIL or scipy


# compute PSNR
def psnr(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return 10 * math.log10(1 / l2_fun(x, y))


# compute RMSE
def rmse(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return math.sqrt(l2_fun(x, y).item() * 3)


# compute SSIM
def ssim(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        return pytorch_ssim.ssim(x, y).item()


# count the number of parameters of a model
def countParameters(model):
    return sum(param.numel() for param in model.parameters())
