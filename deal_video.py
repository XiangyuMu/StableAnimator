
import os
import glob
import random

video_dir = "/data/xuzhenhao/StableAnimator/dataset/video120"  # 视频所在目录
output_dir = "/data/xuzhenhao/StableAnimator/animation_data"  # 图片输出目录

os.makedirs(output_dir, exist_ok=True)

# 获取所有视频文件
videos = glob.glob(os.path.join(video_dir, "*.mp4"))

# 随机抽取 480 个视频作为 rec，剩下的 120 个作为 vec
random.shuffle(videos)
rec_videos = videos[:480]
vec_videos = videos[480:]

# 处理 rec 组
for index, video in enumerate(rec_videos, start=1):
    folder_name = f"{index:05d}"
    output_path = os.path.join(output_dir, "rec", folder_name, "images")
    os.makedirs(output_path, exist_ok=True)

    cmd = f'ffmpeg -i "{video}" -q:v 1 -start_number 0 "{output_path}/frame_%d.png"'
    os.system(cmd)

# 处理 vec 组
for index, video in enumerate(vec_videos, start=1):
    folder_name = f"{index:05d}"
    output_path = os.path.join(output_dir, "vec", folder_name, "images")
    os.makedirs(output_path, exist_ok=True)

    cmd = f'ffmpeg -i "{video}" -q:v 1 -start_number 0 "{output_path}/frame_%d.png"'
    os.system(cmd)
