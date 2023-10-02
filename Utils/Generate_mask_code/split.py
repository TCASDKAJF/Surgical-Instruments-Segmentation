import os
import random
import shutil

directory_path = "G:/Final_Data/segmentation/images"
training_dir = "G:/Final_Data/segmentation/training"
testing_dir = "G:/Final_Data/segmentation/testing"
validation_dir = "G:/Final_Data/segmentation/validation"

# 创建字典以记录存在的video_number
video_numbers = {}
for dir_name in os.listdir(directory_path):
    if dir_name.startswith("video"):
        number = dir_name.split("_")[0][5:]
        video_numbers.setdefault(number, []).append(dir_name)

# 从存在的video_number中随机抽取35个作为训练集，10个作为测试集，5个作为验证集
all_numbers = list(video_numbers.keys())
random.shuffle(all_numbers)

training_numbers = all_numbers[:35]
testing_numbers = all_numbers[35:45]
validation_numbers = all_numbers[45:50]

# 复制训练集文件夹到训练目录
for number in training_numbers:
    for dir_name in video_numbers[number]:
        source_dir = os.path.join(directory_path, dir_name)
        dest_dir = os.path.join(training_dir, dir_name)
        shutil.copytree(source_dir, dest_dir)

# 复制测试集文件夹到测试目录
for number in testing_numbers:
    for dir_name in video_numbers[number]:
        source_dir = os.path.join(directory_path, dir_name)
        dest_dir = os.path.join(testing_dir, dir_name)
        shutil.copytree(source_dir, dest_dir)

# 复制验证集文件夹到验证目录
for number in validation_numbers:
    for dir_name in video_numbers[number]:
        source_dir = os.path.join(directory_path, dir_name)
        dest_dir = os.path.join(validation_dir, dir_name)
        shutil.copytree(source_dir, dest_dir)

