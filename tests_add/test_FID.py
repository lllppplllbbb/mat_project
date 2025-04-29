import os
import sys
import torch
import inspect
from metrics import metric_utils, frechet_inception_distance
import dnnlib
import multiprocessing

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入数据集类
from datasets.dataset_512_val import ImageFolderMaskDataset

def run_fid_test():
    # 设置参数
    data_path = 'F:/MAT_project/MAT/data/test_images'
    seg_path = 'F:/MAT_project/MAT/data/segmentations/test'

    # 打印可用指标
    from metrics.metric_main import list_valid_metrics
    print("可用指标:", list_valid_metrics())

    # 创建数据集对象
    try:
        dataset = ImageFolderMaskDataset(path=data_path, seg_dir=seg_path)
        print(f"成功创建数据集，大小: {len(dataset)}")
        
        # 创建正确的dataset_kwargs
        dataset_kwargs = {
            'dataset': dataset  # 直接传入数据集对象
        }
        
        # 创建选项对象
        print("\n创建选项对象:")
        opts = metric_utils.MetricOptions(dataset_kwargs=dataset_kwargs, cache=False)
        print("成功创建选项对象")
        
        # 添加data和seg_dir属性
        opts.data = data_path
        opts.seg_dir = seg_path
        
        # 确保data_loader_kwargs是一个字典而不是None
        if not hasattr(opts, 'data_loader_kwargs') or opts.data_loader_kwargs is None:
            opts.data_loader_kwargs = {}  # 创建空字典
        
        # 设置data_loader_kwargs的值
        opts.data_loader_kwargs.update({
            'num_workers': 1,
            'pin_memory': True,
            'drop_last': False
        })
        
        print(f"data_loader_kwargs: {opts.data_loader_kwargs}")
        
        # 检查dataset_kwargs是否正确设置
        print(f"dataset_kwargs包含dataset: {'dataset' in opts.dataset_kwargs}")
        print(f"opts.data = {opts.data}")
        print(f"opts.seg_dir = {opts.seg_dir}")
        
        # 修补compute_feature_stats_for_dataset函数
        original_compute_feature_stats = metric_utils.compute_feature_stats_for_dataset
        
        def patched_compute_feature_stats(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
            if data_loader_kwargs is None:
                data_loader_kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
            return original_compute_feature_stats(opts, detector_url, detector_kwargs, rel_lo, rel_hi, batch_size, data_loader_kwargs, max_items, **stats_kwargs)
        
        # 替换原始函数
        metric_utils.compute_feature_stats_for_dataset = patched_compute_feature_stats
        
        # 计算FID
        print("\n开始计算FID...")
        
        # 修改compute_fid函数调用，只计算真实图像的特征
        # 不要尝试解包返回值，它是一个FeatureStats对象
        stats_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts,
            detector_url='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt',
            detector_kwargs={},
            rel_lo=0,
            rel_hi=1,
            batch_size=64,
            max_items=50
        )
        
        # 打印特征统计信息
        print(f"真实图像特征统计信息:")
        print(f"  - 特征数量: {stats_real.num_features}")
        print(f"  - 特征维度: {stats_real.num_items}")
        
        # 可以从stats_real对象中获取均值和协方差
        if hasattr(stats_real, 'mean') and hasattr(stats_real, 'cov'):
            print(f"  - 特征均值形状: {stats_real.mean.shape}")
            print(f"  - 特征协方差矩阵形状: {stats_real.cov.shape}")
        
        # 由于我们没有生成器，所以不能计算完整的FID
        # 但我们可以验证特征提取是否正常工作
        print("\nFID测试完成！成功提取真实图像特征。")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 添加这一行解决多进程问题
    multiprocessing.freeze_support()
    run_fid_test()