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
import random  # 在文件顶部添加这一行

try:
    import pyspng
except ImportError:
    pyspng = None

from datasets.mask_generator_512 import RandomMask

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name, image_dir, mask_dir=None, seg_dir=None,                  # Name of the dataset.
        resolution=512,         # Resolution of the images.
        raw_shape=None,         # Shape of the raw image data (NCHW).
        hole_range=[0,1],       # 修正参数位置并添加逗号
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        **super_kwargs          # 将super_kwargs移到最后
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self.mask_dir = mask_dir if mask_dir else os.path.join(image_dir, 'masks')
        self._resolution = resolution  # 修改为使用_resolution作为私有属性
    

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        idx = idx % len(self._raw_idx)
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        res = self.resolution
        H, W = image.shape[1], image.shape[2]
        h = random.randint(0, H - res) if H > res else 0
        w = random.randint(0, W - res) if W > res else 0
    
        # 加载掩码
        img_filename = os.path.basename(self._image_fnames[self._raw_idx[idx]])
        mask_filename = os.path.splitext(img_filename)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = mask[h:h+res, w:w+res]
        mask = (mask > 128).astype(np.float32)  # 0/255 -> 0/1
        mask = mask[np.newaxis, :, :]  # [1, H, W]
        print(f"[DEBUG] 掩码 {mask_path} 原始值: {np.unique(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))}")
        print(f"[DEBUG] 掩码处理后值: {np.unique(mask)}")
    
        if self._xflip[idx]:
            image = image[:, :, ::-1]
            mask = mask[:, :, ::-1]
    
        # 加载分割图
        if hasattr(self, 'segs') and len(self.segs) > 0:
            seg_idx = self._raw_idx[idx]
            seg_img = cv2.imread(self.segs[seg_idx], cv2.IMREAD_GRAYSCALE)
            if seg_img is None:
                raise ValueError(f"Failed to load segmentation: {self.segs[seg_idx]}")
            if seg_img.shape[0] != H or seg_img.shape[1] != W:
                seg_img = cv2.resize(seg_img, (W, H), interpolation=cv2.INTER_NEAREST)
            seg_img = seg_img[h:h+res, w:w+res]
            seg_img[seg_img == 255] = 0
            seg_img[(seg_img > 20)] = 0
            if idx < 10:
                print(f"[DEBUG] 分割图 {self.segs[seg_idx]} 类别: {np.unique(seg_img)}")
            seg = seg_img[np.newaxis, :, :]
            if self._xflip[idx]:
                seg = seg[:, :, ::-1]
        else:
            seg = np.zeros((1, res, res), dtype=np.float32)
    
        label = self.get_label(idx)
        if label.shape[0] == 0:
            label = np.zeros(20, dtype=np.float32)
    
        return image.copy(), mask, seg, label

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        if hasattr(self, "_resolution") and self._resolution is not None:
            return self._resolution
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]
        
    @resolution.setter
    def resolution(self, value):
        self._resolution = value

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


#----------------------------------------------------------------------------


class ImageFolderMaskDataset(Dataset):
    def __init__(self,
        path,                   
        hole_range=[0,1],
        seg_dir=None,
        mask_dir=None,
        tranform=None,
        resolution=None,
        raw_shape=None,
        name='dataset',         
        **super_kwargs,         
    ):
        self._path = path
        self.mask_dir = mask_dir if mask_dir else os.path.join(path, 'masks')
        self._zipfile = None
        self._hole_range = hole_range

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
        self._seg_dir = seg_dir or os.path.join(path, 'segmentations')
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

            # restricted to 512x512
            # res = 512
            res = self.resolution
            if res is None:
                res = 512  # 设置默认分辨率为512
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
