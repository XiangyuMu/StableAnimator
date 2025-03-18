import shutil
import os
import numpy as np

def select_and_copy_images(src_folder, dest_folder, num_images=20):
    """
    从源文件夹中选择指定数量的图像并复制到目标文件夹
    :param src_folder: 源文件夹路径
    :param dest_folder: 目标文件夹路径
    :param num_images: 选择的图像数量
    """
    try:
        # 检查源文件夹和目标文件夹是否存在
        if not os.path.exists(src_folder):
            print(f"源文件夹 {src_folder} 不存在。")
            return
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # 获取源文件夹中序号为0~100的图像文件
        images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f)) and f.startswith('frame_')]
        images = [f for f in images if 0 <= int(f.split('_')[1].split('.')[0]) <= 100]

        # 如果图像数量不足，直接复制所有图像
        if len(images) <= num_images:
            selected_images = images
        else:
            # 平均选取指定数量的图像
            indices = np.linspace(0, len(images) - 1, num_images, dtype=int)
            selected_images = [images[i] for i in indices]

        # 复制选中的图像到目标文件夹
        for image in selected_images:
            shutil.copy(os.path.join(src_folder, image), os.path.join(dest_folder, image))

        print(f"已从 {src_folder} 中选取 {num_images} 张图像并复制到 {dest_folder} 中。")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 调用函数
select_and_copy_images('/data/muxiangyu/pythonPrograms/StableAnimator/animation_data/00005/clothes_white_complete', '/data/muxiangyu/pythonPrograms/StableAnimator/validation/clothes_white_complete_v5')
select_and_copy_images('/data/muxiangyu/pythonPrograms/StableAnimator/animation_data/00005/heads_white', '/data/muxiangyu/pythonPrograms/StableAnimator/validation/heads_white_v5')
select_and_copy_images('/data/muxiangyu/pythonPrograms/StableAnimator/animation_data/00005/pose_head', '/data/muxiangyu/pythonPrograms/StableAnimator/validation/pose_head_v5')
select_and_copy_images('/data/muxiangyu/pythonPrograms/StableAnimator/animation_data/00005/pose_wo_head', '/data/muxiangyu/pythonPrograms/StableAnimator/validation/pose_wo_head_v5')