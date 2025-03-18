# import os
# import shutil

# vec_folder = "animation_data/vec"
# rec_folder = "animation_data/rec"

# for i in range(1, 121):
#     old_name = os.path.join(vec_folder, f"{i:05d}")
#     new_name = os.path.join(rec_folder, f"{i + 480:05d}")
#     shutil.move(old_name, new_name)

# print("Files renamed and moved successfully.")
import os
import shutil

rec_folder = "animation_data/rec"
destination_folder = "animation_data"

for folder in os.listdir(rec_folder):
    folder_path = os.path.join(rec_folder, folder)
    if os.path.isdir(folder_path):
        shutil.move(folder_path, os.path.join(destination_folder, folder))

print("All folders moved successfully.")
