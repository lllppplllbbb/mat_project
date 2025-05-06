# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import cv2
import json
import torch
import dnnlib
import glob
import random

try:
    import pyspng
except ImportError:
    pyspng = None

from datasets.mask_generator_512 import RandomMask
from datasets.dataset_512 import Dataset

#----------------------------------------------------------------------------

class ImageFolderMaskDataset(Dataset):
    def __init__(self,
        path,                   
        hole_range=[0,1],
        seg_dir=None,
        mask_dir=None,
        tranform=None,
        resolution=512,
        raw_shape=None,
        name='dataset',         
        **super_kwargs,         
    ):
        self._path = path
        self.mask_dir = mask_dir if mask_dir else os.path.join(path, 'masks')
        self._zipfile = None
        self._hole_range = hole_range
        self._seg_dir = seg_dir or os.path.join(path, 'segmentations')

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + [3, resolution, resolution]
        super().__init__(name=name, raw_shape=raw_shape, resolution=resolution, image_dir=path, **super_kwargs)
        self._load_mask(mask_dir)
        self._load_seg()
        
    def _load_mask(self, mpath=None):
        # 添加调试信息
        print(f"[DEBUG] 加载掩码，路径: {mpath if mpath else '默认路径'}")
        # 添加安全检查
        if not hasattr(self, '_path'):
            return
        mpath = mpath or os.path.join(self._path, 'masks')
        self.mask_dir = mpath  # 添加这一行，保存掩码目录路径为类属性
        self.masks = sorted(glob.glob(os.path.join(mpath, '*.png')))
        if len(self.masks) == 0:
            raise IOError(f'No mask files found in {mpath}')   

         
    def _load_seg(self):
        print(f"[DEBUG] 加载分割图，路径: {self._seg_dir}")
        self.segs = []
        for fname in self._image_fnames:
            base_name = os.path.splitext(os.path.basename(fname))[0]
            seg_path = os.path.join(self._seg_dir, f"{base_name}.png")
            if os.path.exists(seg_path):
                self.segs.append(seg_path)
            else:
                raise IOError(f"Segmentation file missing: {seg_path}")
        if len(self.segs) == 0:
            raise IOError(f"No segmentation files found in {self._seg_dir}")
        assert len(self.segs) == len(self._image_fnames), f"Mismatch: {len(self.segs)} segs vs {len(self._image_fnames)} images"
        for seg_path in self.segs[:5]:
            print(f"[DEBUG] 分割图: {seg_path}")

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
            if image.ndim == 2:
                image = image[:, :, np.newaxis] # HW => HWC

            # for grayscale image
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)

            # restricted to resolution
            res = self.resolution
            H, W, C = image.shape
            if H < res or W < res:
                top = 0
                bottom = max(0, res - H)
                left = 0
                right = max(0, res - W)
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
            H, W, C = image.shape
            h = (H - res) // 2
            w = (W - res) // 2
            image = image[h:h+res, w:w+res, :]

            image = np.ascontiguousarray(image.transpose(2, 0, 1)) # HWC => CHW
            return image

    def _load_raw_labels(self):
        fname = 'labels.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels