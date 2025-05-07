# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from losses.pcp import PerceptualLoss
import torchvision.utils as vutils
import os

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        raise NotImplementedError()

class TwoStageLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, truncation_psi=1, pcp_ratio=0.5, sem_ratio=0.3):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.truncation_psi = truncation_psi
        self.pcp = PerceptualLoss(layer_weights={'conv1_2': 1/16, 'conv2_2': 1/8, 'conv3_3': 1/4, 'conv4_3': 1/2, 'conv5_4': 1}).to(device)
        self.pcp_ratio = pcp_ratio
        self.sem_ratio = sem_ratio
        self.output_dir = 'MAT/small_result'
        os.makedirs(self.output_dir, exist_ok=True)
        self.weights = {'adv': 1.0, 'pcp': 10.0, 'sem': 10.0}
        self.target_ratios = {'adv': 0.50, 'pcp': 0.25, 'sem': 0.25}

    def run_G(self, img_in, mask_in, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c, truncation_psi=self.truncation_psi)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, truncation_psi=self.truncation_psi, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img, img_stg1 = self.G_synthesis(img_in, mask_in, ws, return_stg1=True)
        return img, ws, img_stg1

    def run_D(self, img, mask, img_stg1, c, sync):
        with misc.ddp_sync(self.D, sync):
            logits, logits_stg1 = self.D(img, mask, img_stg1, c)
        return logits, logits_stg1

    def perceptual_loss_with_weights(self, pred, target, seg):
        if seg is None:
            print("[WARNING] 分割图为 None，使用默认感知损失")
            return self.pcp(pred, target)[0]
        weights = generate_weight_map(seg)
        pcp_loss, pcp_features = self.pcp(pred, target, seg)
        gt_features = self.pcp(target, target, seg)[1]
        weighted_pcp_loss = 0
        for layer, feat in pcp_features.items():
            feat_size = feat.shape[2:]
            weights_resized = F.interpolate(weights, size=feat_size, mode='nearest')
            layer_loss = F.mse_loss(feat, gt_features[layer], reduction='none') / 100
            weighted_layer_loss = (layer_loss * weights_resized).mean()
            weighted_pcp_loss += weighted_layer_loss * self.pcp.layer_weights[layer]
        weighted_pcp_loss = weighted_pcp_loss / 100
        print(f"[DEBUG] 加权感知损失: {weighted_pcp_loss.item():.4f}")
        return weighted_pcp_loss

    def accumulate_gradients(self, phase, real_img, mask, real_c, gen_z, gen_c, sync, gain, seg_map=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, gen_img_stg1 = self.run_G(real_img, mask, gen_z, gen_c, sync=(sync and not do_Gpl))
                gen_logits, gen_logits_stg1 = self.run_D(gen_img, mask, gen_img_stg1, gen_c, sync=False)
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                loss_Gmain_stg1 = torch.nn.functional.softplus(-gen_logits_stg1)
                pcp_loss = self.perceptual_loss_with_weights(gen_img, real_img, seg_map) if seg_map is not None else self.pcp(gen_img, real_img)[0]
                sem_loss = semantic_loss(gen_img, real_img, seg_map) if seg_map is not None else torch.tensor(0.0, device=self.device)

                # 计算对抗损失
                adv_loss = (loss_Gmain + loss_Gmain_stg1).mean() * 1000.0
                # 动态权重计算总损失
                loss_Gmain_all = (self.weights['adv'] * adv_loss + 
                                self.weights['pcp'] * pcp_loss + 
                                self.weights['sem'] * sem_loss)
                
                # 计算实际占比并调整权重
                total_loss_value = (adv_loss.item() * self.weights['adv'] + 
                                   pcp_loss.item() * self.weights['pcp'] + 
                                   sem_loss.item() * self.weights['sem'])
                if total_loss_value > 0:
                    adv_ratio = (adv_loss.item() * self.weights['adv']) / total_loss_value
                    pcp_ratio = (pcp_loss.item() * self.weights['pcp']) / total_loss_value
                    sem_ratio = (sem_loss.item() * self.weights['sem']) / total_loss_value
                    # 优化权重调整：增加学习率
                    lr = 0.2  # 从 0.1 提高到 0.2
                    self.weights['adv'] = self.weights['adv'] * (1 - lr) + lr * (self.target_ratios['adv'] / (adv_ratio + 1e-8))
                    self.weights['pcp'] = self.weights['pcp'] * (1 - lr) + lr * (self.target_ratios['pcp'] / (pcp_ratio + 1e-8))
                    self.weights['sem'] = self.weights['sem'] * (1 - lr) + lr * (self.target_ratios['sem'] / (sem_ratio + 1e-8))
                    # 限制权重范围
                    self.weights['adv'] = min(max(self.weights['adv'], 0.1), 100.0)
                    self.weights['pcp'] = min(max(self.weights['pcp'], 0.1), 100.0)
                    self.weights['sem'] = min(max(self.weights['sem'], 0.1), 100.0)
                else:
                    adv_ratio, pcp_ratio, sem_ratio = 0, 0, 0

                # 重新计算总损失
                loss_Gmain_all = (self.weights['adv'] * adv_loss + 
                                self.weights['pcp'] * pcp_loss + 
                                self.weights['sem'] * sem_loss)

                # 调试信息
                print(f"[DEBUG] 对抗损失: {adv_loss.item():.4f}, 感知损失: {pcp_loss.item():.4f}, 语义损失: {sem_loss.item():.4f}")
                print(f"[DEBUG] 实际占比: adv={adv_ratio:.2%}, pcp={pcp_ratio:.2%}, sem={sem_ratio:.2%}")
                print(f"[DEBUG] 调整后权重: adv={self.weights['adv']:.2f}, pcp={self.weights['pcp']:.2f}, sem={self.weights['sem']:.2f}")

                # 保存生成的图像
                for i in range(gen_img.size(0)):
                    img_name = f"pred_{real_c[i]}.png" if real_c is not None else f"pred_{i}.png"
                    vutils.save_image(gen_img[i], os.path.join(self.output_dir, img_name), normalize=True, value_range=(-1, 1))

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain_all.mean().mul(gain).backward()

        loss_Dgen = 0
        loss_Dgen_stg1 = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, gen_img_stg1 = self.run_G(real_img, mask, gen_z, gen_c, sync=False)
                gen_logits, gen_logits_stg1 = self.run_D(gen_img, mask, gen_img_stg1, gen_c, sync=False)
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
                loss_Dgen_stg1 = torch.nn.functional.softplus(gen_logits_stg1)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen_all = loss_Dgen + loss_Dgen_stg1
                loss_Dgen_all.mean().mul(gain).backward()

        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                mask_tmp = mask.detach().requires_grad_(do_Dr1)
                real_img_tmp_stg1 = real_img.detach().requires_grad_(do_Dr1)
                real_logits, real_logits_stg1 = self.run_D(real_img_tmp, mask_tmp, real_img_tmp_stg1, real_c, sync=sync)
                loss_Dreal = 0
                loss_Dreal_stg1 = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    loss_Dreal_stg1 = torch.nn.functional.softplus(-real_logits_stg1)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                    training_stats.report('Loss/D/loss_s1', loss_Dgen_stg1 + loss_Dreal_stg1)
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum(), real_logits_stg1.sum()], inputs=[real_img_tmp, mask_tmp], create_graph=True, only_inputs=True)
                        r1_grads_img, r1_grads_mask = r1_grads
                    r1_penalty = r1_grads_img.square().sum([1,2,3]) + r1_grads_mask.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)
            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dreal_stg1 + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------

# 语义损失函数
def generate_weight_map(seg):
    weights = torch.ones_like(seg, dtype=torch.float32) * 1.0  # 背景权重
    weights[seg > 0] = 2.0  # 所有前景
    print(f"[DEBUG] 权重图唯一值: {torch.unique(weights).cpu().numpy()}")  # 将张量移到 CPU 再转 NumPy
    return weights

def semantic_loss(pred, target, seg):
    weights = generate_weight_map(seg)
    loss = F.mse_loss(pred, target, reduction='none') / 100
    weighted_loss = (loss * weights).mean()
    print(f"[DEBUG] 语义损失（归一化后）: {weighted_loss.item():.4f}")
    return weighted_loss