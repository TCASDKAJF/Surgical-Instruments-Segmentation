import cv2
import numpy as np
import os

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"): # 如果你的图像是其他格式，请在这里添加
            img = cv2.imread(os.path.join(input_dir, filename))

            # 将图像恢复到原始比例
            height, width = img.shape[:2]
            resized_img = cv2.resize(img, (int(width*480/256), height), interpolation = cv2.INTER_AREA)
            
            # 中间裁剪出256*256的图像
            center = (int(resized_img.shape[1]/2), int(resized_img.shape[0]/2))
            w_start = center[0]-128
            w_end = center[0]+128
            h_start = center[1]-128
            h_end = center[1]+128
            
            cropped_img = resized_img[h_start:h_end, w_start:w_end]
            
            # 保存处理后的图像
            cv2.imwrite(os.path.join(output_dir, filename), cropped_img)

input_dir = "/root/autodl-tmp/stylegan3/sample" # 输入你的输入文件夹路径
output_dir = "/root/autodl-tmp/stylegan3/processed_sample" # 输入你的输出文件夹路径

process_images(input_dir, output_dir)
