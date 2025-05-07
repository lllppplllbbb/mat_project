# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
_metric_dict = dict() 
import os
import time
import json
import torch
import dnnlib

from . import metric_utils
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import precision_recall
from . import perceptual_path_length
from . import inception_score
from . import psnr_ssim_l1
from . import inception_discriminative_score
import logging
logging.basicConfig(filename='fid_debug.log', level=logging.DEBUG, filemode='w')

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    # assert is_valid_metric(metric)
    # opts = metric_utils.MetricOptions(**kwargs)
    try:
        opts = metric_utils.MetricOptions(cache=False, verbose=True, **kwargs)
        logging.debug(f"计算指标: {metric}, 数据路径: {opts.data}, 分割路径: {opts.seg_dir}")
        print(f"[DEBUG] 计算指标: {metric}, 数据路径: {opts.data}, 分割路径: {opts.seg_dir}")
        dataset = opts.dataset_kwargs['dataset']
        logging.debug(f"数据集大小: {len(dataset)}")
        print(f"[DEBUG] 数据集大小: {len(dataset)}")
        start_time = time.time()
        results = _metric_dict[metric](opts)
        total_time = time.time() - start_time
        logging.debug(f"指标 {metric} 结果: {results}")
        print(f"[DEBUG] 指标 {metric} 结果: {results}")
    except Exception as e:
        logging.error(f"指标 {metric} 失败: {str(e)}")
        print(f"[ERROR] 指标 {metric} 失败: {str(e)}")
        raise

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------
# Primary metrics.

@register_metric
def fid2993_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=2993, num_gen=2993)
    return dict(fid2993_full=fid)

@register_metric
def fid36k5_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=36500, num_gen=36500)
    return dict(fid36k5_full=fid)

@register_metric
def fid_places(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=36500, num_gen=36500)
    return dict(fid36k5_full=fid)

@register_metric
def ids_places(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    u_ids, p_ids = inception_discriminative_score.compute_ids(opts, max_real=36500, num_gen=36500)
    return dict(u_ids=u_ids, p_ids=p_ids)

@register_metric
def psnr36k5_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    psnr, ssim, l1 = psnr_ssim_l1.compute_psnr(opts, max_real=36500)
    return dict(psnr=psnr, ssim=ssim, l1=l1)

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)

@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts, max_real=1000000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k_full=kid)

@register_metric
def pr50k3_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    precision, recall = precision_recall.compute_pr(opts, max_real=200000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_full_precision=precision, pr50k3_full_recall=recall)

@register_metric
def ppl2_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, batch_size=2)
    return dict(ppl2_wend=ppl)

@register_metric
def is50k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=50000, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)

#----------------------------------------------------------------------------
# Legacy metrics.

@register_metric
def fid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=50000, num_gen=50000)
    return dict(fid50k=fid)

@register_metric
def kid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(opts, max_real=50000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k=kid)

@register_metric
def pr50k3(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall = precision_recall.compute_pr(opts, max_real=50000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_precision=precision, pr50k3_recall=recall)

@register_metric
def ppl_zfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='z', sampling='full', crop=True, batch_size=2)
    return dict(ppl_zfull=ppl)

@register_metric
def ppl_wfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='full', crop=True, batch_size=2)
    return dict(ppl_wfull=ppl)

@register_metric
def ppl_zend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='z', sampling='end', crop=True, batch_size=2)
    return dict(ppl_zend=ppl)

@register_metric
def ppl_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=True, batch_size=2)
    return dict(ppl_wend=ppl)

    
#针对小数量数据集
@register_metric
def fid50_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=50, num_gen=50)
    return dict(fid50_full=fid)

@register_metric
def fid10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=10000, num_gen=10000)
    return dict(fid10k_full=fid)

@register_metric
def fid1k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=1000, num_gen=1000)
    return dict(fid1k_full=fid)

@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts, max_real=1000000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k_full=kid)

@register_metric
def pr50k3(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall = precision_recall.compute_pr(opts, max_real=50000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_precision=precision, pr50k3_recall=recall)

@register_metric
def ppl_zend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='z', sampling='end', crop=True, batch_size=2)
    return dict(ppl_zend=ppl)

@register_metric
def ppl_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=True, batch_size=2)
    return dict(ppl_wend=ppl)

#----------------------------------------------------------------------------
