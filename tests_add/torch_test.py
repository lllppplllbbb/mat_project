import torch
import platform
import sys

# 基本信息
print(f"Python版本: {platform.python_version()}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# CUDA详细信息
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA设备属性: {torch.cuda.get_device_properties(0)}")

# 测试CUDA张量操作
if torch.cuda.is_available():
    try:
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print("CUDA张量操作测试成功")
    except Exception as e:
        print(f"CUDA张量操作测试失败: {e}")