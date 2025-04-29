- 激活虚拟环境
```bash
mat_env\Scripts\activate
```


- 生成文件夹下文件名称
```bash
Get-ChildItem -Path f:\MAT_project -Recurse -File | Where-Object { $_.FullName -notmatch '\\mat_env\\' -and $_.FullName -notmatch '\\data\\' } | Select-Object -ExpandProperty FullName > project_framework.txt
```

- 更新requirements.txt
```bash
pip freeze > requirements.txt
```

- 清除pycache
```powershell
Get-ChildItem "F:\MAT_project" -Include "*.pyc","__pycache__" -Recurse | Remove-Item -Force -Recurse
```

```cmd
cmd /c "del /s /q F:\MAT_project\*.pyc && for /d /r F:\MAT_project\ %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d""
```

- mask_generator_512.py
  - --img_dir (必需): 图像目录路径
  - --mask_dir (可选): 掩码保存目录路径，默认为img_dir/masks
  - --resolution (可选): 掩码分辨率，默认为512
```bash
& f:/MAT_project/mat_env/Scripts/python.exe f:/MAT_project/datasets/mask_generator_512_small.py --img_dir "F:/MAT_project/MAT/data/train_images"
& f:/MAT_project/mat_env/Scripts/python.exe f:/MAT_project/datasets/mask_generator_512_small.py --img_dir "F:/MAT_project/MAT/data/test_images" 
```

- 训练文件说明（train.py）
```
--data：训练数据集目录（必填），对应图像文件夹（如F:/MAT_project/MAT/data/train_images）。
--data_val：验证数据集目录（可选，默认等于--data）。
--outdir：输出目录（必填），存训练日志和补全图。
--gpus：GPU数量（默认1，适合你的GTX 1650）。
--batch：批量大小（默认依--cfg）。
--cfg：配置（默认auto，可选places256、places512等）。
--resolution：没有直接的--resolution参数，分辨率由数据集决定（dataset_512.py）。
--mask_root：没有此参数，掩码路径可能由dataset_512.py内部处理。
```



- kaggle的数据钩子
```python 
import os
import sys
import functools

# 原始数据集路径
ORIGINAL_PATH = "/kaggle/input/pascal-voc-mat"
# 新的数据集路径
NEW_PATH = "F:/your_dataset_path"  # 请替换为您的实际数据集路径

def modify_dataset_path(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 修改训练数据集路径
        if 'data' in kwargs and kwargs['data'] is not None:
            if ORIGINAL_PATH in kwargs['data']:
                kwargs['data'] = kwargs['data'].replace(ORIGINAL_PATH, NEW_PATH)
                print(f"已修改训练数据集路径: {kwargs['data']}")
        
        # 修改验证数据集路径
        if 'data_val' in kwargs and kwargs['data_val'] is not None:
            if ORIGINAL_PATH in kwargs['data_val']:
                kwargs['data_val'] = kwargs['data_val'].replace(ORIGINAL_PATH, NEW_PATH)
                print(f"已修改验证数据集路径: {kwargs['data_val']}")
        
        # 调用原始函数
        return func(*args, **kwargs)
    return wrapper
    ```


- 训练指令
```python
import os
#自动找最新数据集
base_dir = '/kaggle/input/pascal-voc-mat'
latest = sorted(os.listdir(base_dir))[-1]
base = f'{base_dir}/{latest}'
print(f'Using dataset: {base}')
#运行训练
!python /kaggle/working/train.py \
  --outdir=/kaggle/working/output \
  --data={base}/train_images \
  --data_val={base}/test_images \
  --kimg=100 \
  --batch=4
```


- 测试文件遇到找不到文件路径的问题
 ```python
 #添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```