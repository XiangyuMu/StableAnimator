import os

base_folder = "/data/muxiangyu/pythonPrograms/StableAnimator/animation_data"
output_file = "/data/muxiangyu/pythonPrograms/StableAnimator/animation_data/video_path.txt"

with open(output_file, "w") as f:
    for i in range(1, 601):
        folder_path = os.path.join(base_folder, f"{i:05d}")
        f.write(folder_path + "\n")

print(f"File {output_file} has been created.")
