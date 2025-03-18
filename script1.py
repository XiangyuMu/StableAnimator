import os
import subprocess

# 图片所在的主目录（包含多个子文件夹，每个子文件夹对应一个视频）
root_dir = "/data/xuzhenhao/StableAnimator/animation_data"
output_dir = "/data/xuzhenhao/StableAnimator/dataset/train/clothes"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历所有子文件夹
for folder_name in sorted(os.listdir(root_dir)):  # 确保顺序
    folder_path = os.path.join(root_dir, folder_name)
    
    # 确保是目录
    if os.path.isdir(folder_path):
        output_video = os.path.join(output_dir, f"{folder_name}.mp4")  # 输出视频名
        
        # 构造 ffmpeg 命令
        cmd = [
            "ffmpeg", "-framerate", "25", "-i", os.path.join(folder_path,"clothes_white_complete", "frame_%d.png"),
            "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", output_video
        ]
        
        # 执行 ffmpeg 命令
        subprocess.run(cmd, check=True)
        print(f"Converted {folder_name} to {output_video}")

print("所有视频转换完成！")
