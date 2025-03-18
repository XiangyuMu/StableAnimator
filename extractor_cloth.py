# import os
# import cv2
# import numpy as np
# def extractClothes(image_path,parse_path,save_path):
#     img = cv2.imread(image_path)
#     parse = cv2.imread(parse_path,cv2.IMREAD_GRAYSCALE)
#     h,w,c = img.shape
#     # 将掩码值改为 0/255
#     clothes_mask = ((parse==59) | (parse==75) | (parse==126) | (parse==140) | (parse==176) | (parse==178) | (parse==194) | (parse==210)).astype(np.uint8) * 255
#     # 扩展维度并应用掩码
#     clothes_mask = clothes_mask[:,:,np.newaxis]
#     img = cv2.bitwise_and(img, img, mask=clothes_mask)
    
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     cv2.imwrite(save_path,img)
    
    

# body_images_path = '/data/xuzhenhao/StableAnimator/inference/case-22-renou/target'
# save_path = '/data/xuzhenhao/StableAnimator/inference/case-22-renou/clothes1'
# parse_images_path = '/data/xuzhenhao/StableAnimator/inference/case-22-renou/mask'

# # for folder_name in tqdm(os.listdir(body_images_path)):
# #     folder_name_path = os.path.join(body_images_path, folder_name)
# #     if os.path.isdir(folder_name_path):
# for image_name in os.listdir(body_images_path):
#     image_path = os.path.join(body_images_path,image_name)
#     parse_path = os.path.join(parse_images_path,image_name+'.png')
#     save_image_path = os.path.join(save_path,image_name)
#     extractClothes(image_path, parse_path, save_image_path)
# import os
# import cv2
# import numpy as np

# # 读取语义分割图像
# parse_path = '/data/xuzhenhao/StableAnimator/inference/case-22-renou/000.png.png'  # 替换为你的实际路径
# parse = cv2.imread(parse_path, cv2.IMREAD_GRAYSCALE)

# # 获取图像尺寸
# h, w = parse.shape

# # 创建一个集合来存储所有唯一的标签值
# unique_labels = set()

# # 遍历图像的每个像素
# print("图像尺寸:", h, "x", w)
# print("\n所有出现的标签值:")
# for i in range(h):
#     for j in range(w):
#         label = parse[i,j]
#         unique_labels.add(label)

# # 按顺序打印所有唯一的标签值
# print("\n按顺序排列的唯一标签值:")
# for label in sorted(unique_labels):
#     print(f"标签值: {label}")

# import numpy as np
# from PIL import Image

# # 语义分割图像路径
# parsing_path = "/data/xuzhenhao/StableAnimator/inference/case-22-renou/frame_44.png.png"

# # 读取语义分割图像（灰度模式）
# parsing_img = Image.open(parsing_path).convert('L')
# parsing_array = np.array(parsing_img)

# # 获取所有唯一的标签值
# unique_labels = np.unique(parsing_array)

# # 输出标签值
# print("该语义分割图的标签值:", unique_labels)

import numpy as np
from PIL import Image

# 读取语义分割图像（不转换灰度）
parsing_path = "/data/xuzhenhao/StableAnimator/inference/case-23/81FyMPk-WIS/000.png_gray.png"
parsing_img = Image.open(parsing_path)
parsing_array = np.array(parsing_img)  # 直接转换为 numpy 数组

# 检查数据类型（是否为 uint8）
print("数据类型:", parsing_array.dtype)

# 获取唯一标签值
unique_labels = np.unique(parsing_array)
print("该语义分割图的标签值:", unique_labels)

