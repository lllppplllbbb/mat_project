import numpy as np
from PIL import Image, ImageDraw
import math
import random
import os
import argparse

def RandomBrush(
    max_tries,
    s,
    min_num_vertex=4,
    max_num_vertex=18,
    mean_angle=2*math.pi / 5,
    angle_range=2*math.pi / 15,
    min_width=12,
    max_width=48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(3 * coef), s // 2)
        MultiFill(int(2 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(4 * coef), s))  # hole denoted as 0, reserved as 1
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

def BatchRandomMask(batch_size, s, hole_range=[0, 1]):
    return np.stack([RandomMask(s, hole_range=hole_range) for _ in range(batch_size)], axis=0)

def save_masks(img_dir, mask_dir=None, resolution=128):
    # 如果没有指定mask_dir，则默认在img_dir下创建masks子目录
    if mask_dir is None:
        mask_dir = os.path.join(img_dir, "masks")
    
    os.makedirs(mask_dir, exist_ok=True)
    
    for img_name in os.listdir(img_dir):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            mask = RandomMask(s=resolution)
            mask = mask[0] * 255  # [1,s,s] -> [s,s], 0/1 -> 0/255
            mask_img = Image.fromarray(mask.astype(np.uint8), mode='L')
            mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + '.png')
            mask_img.save(mask_path)
            print(f"Saved {mask_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help='图像目录路径')
    parser.add_argument('--mask_dir', type=str, help='掩码保存目录路径，默认为img_dir/masks')
    parser.add_argument('--resolution', type=int, default=512, help='掩码分辨率')
    args = parser.parse_args()

    # 生成并保存掩码
    save_masks(args.img_dir, args.mask_dir, args.resolution)

    # 原统计代码
    res = args.resolution
    cnt = 2000
    tot = 0
    for i in range(cnt):
        mask = RandomMask(s=res)
        tot += mask.mean()
    print(f"Average mask value: {tot / cnt}")