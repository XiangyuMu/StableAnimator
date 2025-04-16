# import shutil
# import os
# import numpy as np

# def select_and_copy_images(src_folder, dest_folder, num_images=20):
#     """
#     从源文件夹中选择指定数量的图像并复制到目标文件夹
#     :param src_folder: 源文件夹路径
#     :param dest_folder: 目标文件夹路径
#     :param num_images: 选择的图像数量
#     """
#     try:
#         # 检查源文件夹和目标文件夹是否存在
#         if not os.path.exists(src_folder):
#             print(f"源文件夹 {src_folder} 不存在。")
#             return
#         if not os.path.exists(dest_folder):
#             os.makedirs(dest_folder)

#         # 获取源文件夹中序号为0~100的图像文件
#         images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f)) and f.startswith('frame_')]
#         images = [f for f in images if 0 <= int(f.split('_')[1].split('.')[0]) <= 100]

#         # 如果图像数量不足，直接复制所有图像
#         if len(images) <= num_images:
#             selected_images = images
#         else:
#             # 平均选取指定数量的图像
#             indices = np.linspace(0, len(images) - 1, num_images, dtype=int)
#             selected_images = [images[i] for i in indices]

#         # 复制选中的图像到目标文件夹
#         for image in selected_images:
#             shutil.copy(os.path.join(src_folder, image), os.path.join(dest_folder, image))

#         print(f"已从 {src_folder} 中选取 {num_images} 张图像并复制到 {dest_folder} 中。")
#     except Exception as e:
#         print(f"处理过程中出现错误: {e}")

# # 调用函数
# select_and_copy_images('/data/muxiangyu/pythonPrograms/StableAnimator/animation_data/00005/clothes_white_complete', '/data/muxiangyu/pythonPrograms/StableAnimator/validation/clothes_white_complete_v5')
# select_and_copy_images('/data/muxiangyu/pythonPrograms/StableAnimator/animation_data/00005/centered_heads', '/data/muxiangyu/pythonPrograms/StableAnimator/validation/centered_heads_v5')
# select_and_copy_images('/data/muxiangyu/pythonPrograms/StableAnimator/animation_data/00005/pose_head_new', '/data/muxiangyu/pythonPrograms/StableAnimator/validation/pose_head_new_v5')
# select_and_copy_images('/data/muxiangyu/pythonPrograms/StableAnimator/animation_data/00005/pose_wo_head', '/data/muxiangyu/pythonPrograms/StableAnimator/validation/pose_wo_head_v5')


# from PIL import Image
# import os
# from tqdm import tqdm

# def crop_images_to_512x256(src_base_path, dest_base_path):
#     """
#     将源路径下的子文件夹中的二级子文件夹内的图像从 512x512 剪切为 512x256，并保存到目标路径下
#     :param src_base_path: 源基础路径
#     :param dest_base_path: 目标基础路径
#     """
#     try:
#         # 获取所有子文件夹
#         subfolders = [os.path.join(src_base_path, name) for name in os.listdir(src_base_path)
#                       if os.path.isdir(os.path.join(src_base_path, name))]

#         for subfolder in tqdm(subfolders):
#             # 获取子文件夹中的二级子文件夹
#             inner_subfolders = [os.path.join(subfolder, name) for name in os.listdir(subfolder)
#                                 if os.path.isdir(os.path.join(subfolder, name))]
            
#             for inner_subfolder in inner_subfolders:
#                 # 获取所有图像文件
#                 images = [f for f in os.listdir(inner_subfolder) if os.path.isfile(os.path.join(inner_subfolder, f))]
                
#                 # 创建目标文件夹
#                 relative_path = os.path.relpath(inner_subfolder, src_base_path)
#                 dest_folder = os.path.join(dest_base_path, relative_path)
#                 os.makedirs(dest_folder, exist_ok=True)

#                 for image in images:
#                     src_image_path = os.path.join(inner_subfolder, image)
#                     dest_image_path = os.path.join(dest_folder, image)

#                     if os.path.exists(dest_image_path):
#                         continue
#                     if not image.endswith('.png'):
#                         print(f"图像 {src_image_path} 不是 PNG 格式，跳过裁剪。")
#                         continue

#                     # 打开图像并裁剪
#                     with Image.open(src_image_path) as img:
#                         if img.size == (512, 512):
#                             cropped_img = img.crop((128, 0, 384, 512))  # 保留中间部分
#                             cropped_img.save(dest_image_path)
#                         else:
#                             print(f"图像 {src_image_path} 的大小不是 512x512，跳过裁剪。")

#         print(f"已完成将 {src_base_path} 下的图像裁剪并保存到 {dest_base_path}。")
#     except Exception as e:
#         print(f"处理过程中出现错误: {e}")
#         print(src_image_path)

# # 调用函数
# src_base_path = '/data/muxiangyu/pythonPrograms/StableAnimator/animation_data/video_512_256'
# dest_base_path = '/data/muxiangyu/pythonPrograms/StableAnimator/animation_data_crop'
# crop_images_to_512x256(src_base_path, dest_base_path)

from PIL import Image
import os
from tqdm import tqdm

def downsample_images(src_base_path, dest_base_path):
    """
    将源路径下的子文件夹中的二级子文件夹内的图像从 512x256 下采样到 256x128，并保存到目标路径下
    :param src_base_path: 源基础路径
    :param dest_base_path: 目标基础路径
    """
    try:
        # 获取所有子文件夹
        subfolders = [os.path.join(src_base_path, name) for name in os.listdir(src_base_path)
                      if os.path.isdir(os.path.join(src_base_path, name))]

        for subfolder in tqdm(subfolders, desc="Processing subfolders"):
            # 获取子文件夹中的二级子文件夹
            inner_subfolders = [os.path.join(subfolder, name) for name in os.listdir(subfolder)
                                if os.path.isdir(os.path.join(subfolder, name))]
            
            for inner_subfolder in inner_subfolders:
                # 获取所有图像文件
                images = [f for f in os.listdir(inner_subfolder) if os.path.isfile(os.path.join(inner_subfolder, f))]
                
                # 创建目标文件夹
                relative_path = os.path.relpath(inner_subfolder, src_base_path)
                dest_folder = os.path.join(dest_base_path, relative_path)
                os.makedirs(dest_folder, exist_ok=True)

                for image in images:
                    src_image_path = os.path.join(inner_subfolder, image)
                    dest_image_path = os.path.join(dest_folder, image)

                    if os.path.exists(dest_image_path):
                        continue
                    if not image.endswith('.png'):
                        print(f"图像 {src_image_path} 不是 PNG 格式，跳过下采样。")
                        continue

                    # 打开图像并下采样
                    with Image.open(src_image_path) as img:
                        if img.size == (256, 512):
                            downsampled_img = img.resize((128, 256), Image.Resampling.LANCZOS)
                            downsampled_img.save(dest_image_path)
                        else:
                            print(f"图像 {src_image_path} 的大小不是 512x256，跳过下采样。")

        print(f"已完成将 {src_base_path} 下的图像下采样并保存到 {dest_base_path}。")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 调用函数、
base_path = '/data/muxiangyu/pythonPrograms/StableAnimator/animation_data'  # 替换为实际的基础路径
new_folder_name = 'video600_512_256'
src_base_path = os.path.join(base_path, new_folder_name)
dest_base_path = os.path.join(base_path, 'video600_256_128')
downsample_images(src_base_path, dest_base_path)