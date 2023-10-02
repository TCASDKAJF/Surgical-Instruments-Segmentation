# import os
# import random
# import matplotlib.pyplot as plt
# from PIL import Image
#
# # 定义文件夹路径
# folders = {
#     "htc": "G:\\baseline\\result\\mmdet_old_data\\htc\\new_htc_result\\vis",
#     "resnet": "G:\\baseline\\result\\mmdet_old_data\\maskrcnn_resnet\\new_resnet_result\\vis",
#     "mine": r'G:\baseline\result\pytorch\results_threshold_0.4\results_threshold_0.4',
#     "MSMA": r'G:\baseline\result\pytorch\MSMA_results\newnewpre',
#     "yolo": r'G:\baseline\result\yolo\yolov5-seg',
#     "groundtruth": "G:\\baseline\\result\\matched_images"
# }
#
#
# # 从任意文件夹获取所有图片名称
# sample_images = [f for f in os.listdir(folders["htc"]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
# selected_images = []
#
# # 随机选择6张图片，确保每张图片在所有文件夹中都存在
# while len(selected_images) < 6:
#     img_name = random.choice(sample_images)
#     if img_name not in selected_images and all(os.path.exists(os.path.join(folder_path, img_name)) for folder_path in folders.values()):
#         selected_images.append(img_name)
#
# # 其他部分代码不变
#
#
# # 为每个文件夹和选定的图片创建子图
# fig, axs = plt.subplots(6, len(folders), figsize=(15, 15))
#
# # 遍历每个文件夹和选定的图片
# for i, img_name in enumerate(selected_images):
#     for j, (folder_name, folder_path) in enumerate(folders.items()):
#         img_path = os.path.join(folder_path, img_name)
#         if os.path.exists(img_path):
#             img = Image.open(img_path)
#             axs[i, j].imshow(img)
#             axs[i, j].set_title(f"{folder_name} ({img_name})")
#             axs[i, j].axis('off')
#         else:
#             axs[i, j].set_title(f"{folder_name} (image_not_found)")
#             axs[i, j].axis('off')
#
# plt.tight_layout()
from PIL import Image, ImageDraw, ImageFont
import os

# 文件夹路径和图像名
folders = {
    "htc": "G:\\baseline\\result\\mmdet_old_data\\htc\\new_htc_result\\vis",
    "yolo": "G:\\baseline\\result\\yolo\\yolov5-seg",
    "resnet": "G:\\baseline\\result\\mmdet_old_data\\maskrcnn_resnet\\new_resnet_result\\vis",
    "ours": "G:\\baseline\\result\\pytorch\\results_threshold_0.4\\results_threshold_0.4",
    "CME": r"G:\baseline\result\pytorch\MSMA_results\once",
    "groundtruth": "G:\\baseline\\result\\matched_images",
    "Image": "G:\\baseline\\Swin\\polyp-seg\\all_images"}

img_names = [
    "video06_clip03_step5_054.jpg",
    "video06_clip04_step7_009.jpg",
    "video13_clip05_step5_067.jpg",
    "video19_clip07_step5_111.jpg",
    "video14_clip08_step5_001.jpg",
    "video16_clip06_step4_091.jpg",
    "video13_clip06_step6_038.jpg",
    "video01_clip01_step1_024.jpg",
    "video03_clip01_step1_085.jpg",
    "video14_clip07_step3_018.jpg",
]

# 使用第一个图像来确定单个图像的尺寸
sample_img_path = os.path.join(list(folders.values())[0], img_names[0])
sample_img = Image.open(sample_img_path)
width, height = sample_img.size

# 添加标题行的高度
title_height = 150
line_width = 10  # 设置白线的宽度

# 创建一个足够大的新图像
new_img_width = (width + line_width) * len(folders) - line_width
new_img_height = title_height + (height + line_width) * len(img_names) - line_width
new_img = Image.new("RGB", (new_img_width, new_img_height), "white")

# 绘制类别名称
draw = ImageDraw.Draw(new_img)
font = ImageFont.truetype("arial.ttf", size=60)

# 拼接图像
for i, img_name in enumerate(img_names):
    for j, (folder_name, folder_path) in enumerate(folders.items()):
        img_path = os.path.join(folder_path, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
        else:
            img = Image.new("RGB", (width, height), "white")

        x_offset = j * (width + line_width)
        y_offset = title_height + i * (height + line_width)
        new_img.paste(img, (x_offset, y_offset))

        # 在整个图像的最上方添加类别名称
        if i == 0:
            text_bbox = draw.textbbox((0, 0), folder_name, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text(((j * width) + (width - text_width) // 2, (title_height - text_height) // 2), folder_name, font=font, fill="black")

# 保存到指定目录
output_path = "G:\\baseline\\result\\DATA_RESULTS_FOR_GRAPH\\merged_image.jpg"
new_img.save(output_path)
print(f"图像已保存至：{output_path}")

# 保存或显示新图像
new_img.show()