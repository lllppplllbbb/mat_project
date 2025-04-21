- 激活虚拟环境
```bash
mat_env\Scripts\activate
```
- 生成文件夹下文件名称
```bash
Get-ChildItem -Path f:\MAT_project -Recurse -File | Where-Object { $_.FullName -notmatch '\\mat_env\\' -and $_.FullName -notmatch '\\Blog\\' } | Select-Object -ExpandProperty FullName > project_framework.txt
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