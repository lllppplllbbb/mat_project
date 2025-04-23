import os
import zipfile
import datetime

def create_images_zip(train_dir, test_dir, seg_dir=None, output_zip=None):
    """
    将训练和测试图像文件以及分割图打包成zip压缩包
    
    参数:
        train_dir: 训练图像目录路径
        test_dir: 测试图像目录路径
        seg_dir: 分割图目录路径（可选）
        output_zip: 输出的zip文件名，如果为None则使用当前日期作为文件名
    """
    # 如果没有指定输出文件名，使用当前日期
    if output_zip is None:
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        output_zip = f"MAT_images_{current_date}.zip"
    
    # 确保输出路径是绝对路径
    if not os.path.isabs(output_zip):
        output_zip = os.path.join(os.path.dirname(train_dir), output_zip)
    
    print(f"正在创建压缩包: {output_zip}")
    
    # 创建zip文件
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 添加训练图像
        if os.path.exists(train_dir):
            for root, dirs, files in os.walk(train_dir):
                # 获取相对路径
                rel_path = os.path.relpath(root, os.path.dirname(train_dir))
                
                # 添加文件到zip
                for file in files:
                    file_path = os.path.join(root, file)
                    # 在zip中保持目录结构
                    zipf.write(file_path, os.path.join(rel_path, file))
                    print(f"添加文件: {os.path.join(rel_path, file)}")
        else:
            print(f"警告: 训练图像目录 {train_dir} 不存在")
        
        # 添加测试图像
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                # 获取相对路径
                rel_path = os.path.relpath(root, os.path.dirname(test_dir))
                
                # 添加文件到zip
                for file in files:
                    file_path = os.path.join(root, file)
                    # 在zip中保持目录结构
                    zipf.write(file_path, os.path.join(rel_path, file))
                    print(f"添加文件: {os.path.join(rel_path, file)}")
        else:
            print(f"警告: 测试图像目录 {test_dir} 不存在")
        
        # 添加分割图（如果提供）
        if seg_dir and os.path.exists(seg_dir):
            # 添加训练分割图
            train_seg_dir = os.path.join(seg_dir, "train")
            if os.path.exists(train_seg_dir):
                for root, dirs, files in os.walk(train_seg_dir):
                    # 获取相对路径
                    rel_path = os.path.relpath(root, os.path.dirname(seg_dir))
                    
                    # 添加文件到zip
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 在zip中保持目录结构
                        zipf.write(file_path, os.path.join(rel_path, file))
                        print(f"添加文件: {os.path.join(rel_path, file)}")
            else:
                print(f"警告: 训练分割图目录 {train_seg_dir} 不存在")
                
            # 添加测试分割图
            test_seg_dir = os.path.join(seg_dir, "test")
            if os.path.exists(test_seg_dir):
                for root, dirs, files in os.walk(test_seg_dir):
                    # 获取相对路径
                    rel_path = os.path.relpath(root, os.path.dirname(seg_dir))
                    
                    # 添加文件到zip
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 在zip中保持目录结构
                        zipf.write(file_path, os.path.join(rel_path, file))
                        print(f"添加文件: {os.path.join(rel_path, file)}")
            else:
                print(f"警告: 测试分割图目录 {test_seg_dir} 不存在")
        elif seg_dir:
            print(f"警告: 分割图目录 {seg_dir} 不存在")
    
    print(f"压缩包创建完成: {output_zip}")
    dirs_included = [train_dir, test_dir]
    if seg_dir:
        dirs_included.append(f"{seg_dir}/train")
        dirs_included.append(f"{seg_dir}/test")
    print(f"包含的目录: {', '.join(dirs_included)}")

if __name__ == "__main__":
    # 目录设置
    train_images_dir = "F:/MAT_project/MAT/data/train_images"
    test_images_dir = "F:/MAT_project/MAT/data/test_images"
    segmentations_dir = "F:/MAT_project/MAT/data/segmentations"
    
    # 创建压缩包
    create_images_zip(train_images_dir, test_images_dir, segmentations_dir)

# 如果需要自定义输出文件名，可以使用以下代码
# if __name__ == "__main__":
#     train_images_dir = "F:/MAT_project/MAT/data/train_images"
#     test_images_dir = "F:/MAT_project/MAT/data/test_images"
#     segmentations_dir = "F:/MAT_project/MAT/data/segmentations"
#     create_images_zip(train_images_dir, test_images_dir, segmentations_dir, "MAT_images.zip")