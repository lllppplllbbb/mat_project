import os
import zipfile
import datetime

def create_zip_archive(source_dir, output_zip=None):
    """
    将项目文件打包成zip压缩包，排除指定的文件和目录
    
    参数:
        source_dir: 源目录路径
        output_zip: 输出的zip文件名，如果为None则使用当前日期作为文件名
    """
    # 要排除的文件和目录
    exclude_list = [
        'Blog', 
        'data', 
        'MAT', 
        'mat_env', 
        'test_sets', 
        'test_add', 
        '.gitignore', 
        'project_framework.txt', 
        'README.md',
        '.git',
        '__pycache__',
    ]
    
    # 如果没有指定输出文件名，使用当前日期
    if output_zip is None:
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        output_zip = f"MAT_project_{current_date}.zip"
    
    # 确保输出路径是绝对路径
    if not os.path.isabs(output_zip):
        output_zip = os.path.join(source_dir, output_zip)
    
    print(f"正在创建压缩包: {output_zip}")
    
    # 创建zip文件
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历源目录
        for root, dirs, files in os.walk(source_dir):
            # 修改dirs列表以排除目录（这会影响os.walk的遍历）
            dirs[:] = [d for d in dirs if d not in exclude_list]
            
            # 获取相对路径
            rel_path = os.path.relpath(root, source_dir)
            if rel_path == '.':
                rel_path = ''
            
            # 添加文件到zip
            for file in files:
                # 跳过排除列表中的文件
                if file in exclude_list:
                    continue
                
                # 跳过输出的zip文件本身
                file_path = os.path.join(root, file)
                if os.path.abspath(file_path) == os.path.abspath(output_zip):
                    continue
                
                # 添加到zip
                zipf.write(file_path, os.path.join(rel_path, file))
                print(f"添加文件: {os.path.join(rel_path, file)}")
    
    print(f"压缩包创建完成: {output_zip}")
    print(f"排除的文件和目录: {', '.join(exclude_list)}")

if __name__ == "__main__":
    # 获取MAT_project目录作为源目录（当前脚本在tests_add子目录中）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)  # 获取父目录，即MAT_project目录
    create_zip_archive(project_dir)

# if __name__ == "__main__":
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     project_dir = os.path.dirname(current_dir)  # 获取父目录，即MAT_project目录
#     create_zip_archive(project_dir, "自定义文件名.zip")