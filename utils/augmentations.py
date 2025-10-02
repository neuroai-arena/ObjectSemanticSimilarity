#!/usr/bin/python
# _____________________________________________________________________________
import math
# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import sys, os
from typing import Optional, List, Tuple

import torch
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

from torchvision.transforms.v2 import Compose

from utils.constants import DATASETS
from kornia import augmentation as TF
import torchvision


class GaussianBlur(object):
    """
    https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/augmentations.py
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    """
    https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/augmentations.py
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def get_transform_list(args, crop_size=None, tensor_normalize=True, normalize=None):
    transformations = []
    if args.min_crop != 1 and not args.one_crop and not args.crop_first:
        if args.image_padding:
            transformations.append(TF.PadTo(args.image_padding))
        transformations.append(get_resized_crop(args, crop_size))
    if args.flip:
        transformations.append(get_flip(args))
    if args.jitter != 0 and not args.unijit:
        transformations.append(get_jitter(args))
    if args.grayscale and not args.unijit:
        transformations.append(get_grayscale(args))
    if args.blur:
        # transformations.append(TF.RandomGaussianBlur(kernel_size=args.blur, sigma=(0.1, 2.0), p=0.2))
        transformations.append(TF.RandomGaussianBlur(kernel_size=args.blur, sigma=(0.1, 2.0), p=args.pblur))
    if args.solarize:
        transformations.append(TF.RandomSolarize(p=args.solarize))
    if tensor_normalize:
        # transformations.append(TF.ToTensor())
        transformations.append(normalize)
    return torch.nn.Sequential(*transformations)

def get_transformations(args, crop_size=None, tensor_normalize=True):
    """
    contrast_type: str 'classic', 'cltt', else
    rgb_mean: tuple of float (r, g, b)
    rgb_std: tuple of float (r, g, b)
    crop_size: int, pixels
    """

    norm_dataset = args.dataset if not args.imgnet_stats else "MVImgNet"
    normalize = TF.Normalize(mean=DATASETS[norm_dataset]['rgb_mean'], std=DATASETS[norm_dataset]['rgb_std'])
    val_transform = normalize

    if args.contrast != 'time' and args.kornia:
        train_transform = get_transform_list(args, crop_size=crop_size, tensor_normalize=tensor_normalize, normalize=normalize)
    else:
        train_transform = val_transform

    # if args.contrast == 'classic':
        # if classic use the TwoContrastTransform to create contrasts
        # train_transform = TwoContrastTransform(train_transform)
    return train_transform, val_transform


class CenterCropAndResize(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, proportion, size):
        self.proportion = proportion
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped and image.
        """
        w, h = (np.array(img.size) * self.proportion).astype(int)
        img = F.resize(
            F.center_crop(img, (h, w)),
            (self.size, self.size)
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(proportion={0}, size={1})'.format(self.proportion, self.size)

def get_resized_crop(args, crop_size):
    ratio = (3.0 / args.crop_ratio, args.crop_ratio / 3.0)
    crop_size = crop_size if crop_size else args.crop_size
    fn = RandomResizedCropWithParams if "resize_crop" in args.aug_params else TF.RandomResizedCrop
    # return transforms.RandomApply([fn(size=crop_size, scale=(args.min_crop, args.max_crop), ratio=ratio)], p=args.pcrop)
    return fn(size=crop_size, scale=(args.min_crop, args.max_crop), ratio=ratio, p=args.pcrop)


def get_jitter(args):
    s = args.jitter_strength
    if "jitter" not in args.aug_params:
        return TF.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s,p=args.jitter)
    return ColorJitterWithParams(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)


def get_grayscale(args):
    return RandomGrayscaleWithParams() if "grayscale" in args.aug_params else TF.RandomGrayscale(p=0.2)


def get_flip(args):
    return RandomHorizontalFlipWithParams() if "flip" in args.aug_params else TF.RandomHorizontalFlip(p=args.flip)


class TwoContrastTransform(torch.nn.Module):
    """
    Create two contrasts of the same image using the given
    torchvision transform
    """

    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def forward(self, x):
        if hasattr(self.transform, "params"):
            t1 = self.transform(x)
            t2 = self.transform(x)
            return [t1, t2, self.transform.prev_params - self.transform.params]
        return [self.transform(x), self.transform(x), torch.zeros(1, 1)]


class ComposeParams(Compose):
    def __init__(self, transform, *args, **kwargs):
        super().__init__(transform)
        self.prev_params = torch.zeros((1,))
        self.params = torch.zeros((1,))

    def augment_img_with_params(self, img):
        params = []
        self.prev_params = self.params
        for t in self.transforms:
            res_t = t(img)
            if isinstance(res_t, tuple):
                img, params_temp = res_t
                params.append(params_temp)
            else:
                img = res_t
        if params:
            self.params = torch.cat(params)
        return img

    def __call__(self, img):
        return self.augment_img_with_params(img)


class ComposedAugsWithParams:
    def __init__(self, removal=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_params = None
        self.params = None
        self.augmentations = [
            RandomResizedCropWithParams(),
            RandomHorizontalFlipWithParams(),
            ColorJitterWithParams(),
            RandomGrayscaleWithParams()
        ]
        if removal is not None:
            self.augmentations.pop(removal)

    def augment_img_with_params(self, img):
        params = []
        for t in self.augmentations:
            res_t = t(img)
            if isinstance(res_t, tuple):
                img, params_temp = res_t
                params.append(params_temp)
        self.prev_params = self.params
        self.params = torch.cat(params)
        return img

    def __call__(self, img):
        return self.augment_img_with_params(img)


class PositionEncoding:
    def __init__(self, img_size, *args, **kwargs):
        self.pos_embedding = torch.stack(
            [torch.linspace(-1, 1, img_size).unsqueeze(0).repeat(img_size, 1),
             torch.linspace(-1, 1, img_size).unsqueeze(1).repeat(1, img_size)]).unsqueeze(0)

    def __call__(self, imgs):
        batch_size = imgs.size(0)
        return torch.cat([imgs, self.pos_embedding.repeat(batch_size, 1, 1, 1).to(imgs.device)], 1)


class RandomCenterCrop:
    def __init__(self, size=224, scale=(0.08, 1.0)):
        self.scale = scale
        self.size = size

    def __call__(self, img):
        crop_shape = int(random.uniform(self.scale[0], self.scale[1]) * self.size)
        out = TF.functional.center_crop(img, [crop_shape, crop_shape])
        out = TF.functional.resize(out, (self.size, self.size))
        return out


class RandomResizedCropWithParams:
    dim = 4

    def __init__(self, size=(128, 128), scale=(0.2, 1.0), ratio=(0.75, 4 / 3)):
        self.scale = scale
        self.ratio = ratio
        self.size = size
        # pm_wh = (scale[1] + scale[0])*size[0] /2
        min_wh = math.sqrt(scale[0]*ratio[0])*size[0]
        max_wh = min(size[0], math.sqrt(scale[1]*ratio[1])*size[1])
        # pm_ij = (size[0] + scale[0]*size[0]) / 2
        min_ij = 0
        max_ij = min(size[0], (size[0] - math.sqrt(scale[0]*ratio[0])*size[0]))
        # self.params_mean = torch.tensor([pm_ij, pm_ij, pm_wh, pm_wh])
        self.params_min = torch.tensor([min_ij, min_ij, min_wh, min_wh])
        self.params_max = torch.tensor([max_ij, max_ij, max_wh, max_wh])
        self.params_minmax = self.params_max - self.params_min

        # self.exp_min = torch.tensor([1000]*4)
        # self.exp_max = torch.tensor([-1000]*4)

    def __call__(self, img):
        params = TF.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        tparams = torch.tensor(params, dtype=torch.float32)
        # self.exp_min = torch.min(torch.cat((tparams.view(1, -1), self.exp_min.view(1,-1)), dim=0), dim=0)[0]
        # self.exp_max = torch.max(torch.cat((tparams.view(1, -1), self.exp_max.view(1,-1)), dim=0), dim=0)[0]
        # print(self.exp_max, self.params_max)
        out = TF.functional.resized_crop(img, *params, size=self.size)
        return out, (tparams - self.params_min)/self.params_minmax -0.5
        # return out, torch.tensor(params, dtype=torch.float32) / self.size[0]


class RandomHorizontalFlipWithParams:
    dim = 1

    def __call__(self, img):
        params = np.round(np.random.rand())
        out = TF.functional.hflip(img) if params > 0.5 else img
        return out, torch.tensor([params-0.5], dtype=torch.float32)



class ColorJitterWithParams(transforms.ColorJitter):
    dim=8
    def __init__(self, *args, p = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.min_params = torch.tensor([0, 0, 0, 0, self.brightness[0], self.contrast[0], self.saturation[0], self.hue[0]])
        self.max_params = torch.tensor([3, 3, 3, 3, self.brightness[1], self.contrast[1], self.saturation[1], self.hue[1]])
        self.minmax_params = self.max_params - self.min_params


    def get_free_params(self):
        return torch.tensor((0,1,2,3,1,1,1,0))

    def get_structured_params(self, img):
        # return torch.tensor((0,1,2,3,1,1,1,0)).view(1,-1)
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return torch.cat((fn_idx.view(-1),torch.tensor([brightness_factor, contrast_factor,
                          saturation_factor, hue_factor]).view(-1)),dim=0)

    def apply_t(self, img, params):
        brightness_factor = params[4]
        contrast_factor = params[5]
        saturation_factor = params[6]
        hue_factor = params[7]
        for i in range(4):
            fn_id = params[i]
            if fn_id == 0:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3:
                img = F.adjust_hue(img, hue_factor)

        return img

    def forward(self, img):
        if np.random.rand() < self.p:
            t_params = self.get_structured_params(img)
            img = self.apply_t(img, t_params)
        else:
            t_params = self.get_free_params()
        return img, (t_params - self.min_params)/self.minmax_params - 0.5

class ColorJitterWithParams2:
    dim = 4

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        if np.random.rand() < 0.8:
            brightness = torch.empty(1).uniform_(1 - self.brightness, 1 + self.brightness)
            contrast = torch.empty(1).uniform_(1 - self.contrast, 1 + self.contrast)
            saturation = torch.empty(1).uniform_(1 - self.saturation, 1 + self.saturation)
            hue = torch.empty(1).uniform_(-self.hue, self.hue)
            fns = [lambda x: TF.functional.adjust_brightness(x, brightness),
                   lambda x: TF.functional.adjust_contrast(x, contrast),
                   lambda x: TF.functional.adjust_saturation(x, saturation),
                   lambda x: TF.functional.adjust_hue(x, hue)]
            random.shuffle(fns)
            for fn in fns:
                img = fn(img)
            params = torch.tensor([brightness, contrast, saturation, hue], dtype=torch.float32)
        else:
            params = torch.tensor([1, 1, 1, 0], dtype=torch.float32)
        return img, params


class RandomGrayscaleWithParams:
    dim = 1

    def __call__(self, img):
        p = np.random.rand()
        out = TF.functional.rgb_to_grayscale(img, num_output_channels=3) if p > 0.8 else img
        return out, torch.tensor([p > 0.8], dtype=torch.float32)

