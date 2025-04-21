- 激活虚拟环境
```bash
mat_env\Scripts\activate
```
- 生成文件夹下文件名称
```bash
Get-ChildItem -Path f:\MAT_project -Recurse -File | Where-Object { $_.FullName -notmatch '\\mat_env\\' -and $_.FullName -notmatch '\\Blog\\' } | Select-Object -ExpandProperty FullName > project_framework.txt
```