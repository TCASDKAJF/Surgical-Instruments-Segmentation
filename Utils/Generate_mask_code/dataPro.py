#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/7/28 11:19
# @Author  : Linxuan Jiang
# @File    : 数据处理.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright MIT

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from glob import glob
from tqdm import tqdm
# 这里就是填一些有关你数据集的信息
INFO = {
    "description": "Example Dataset",
    "url": "none",
    "version": "0.1.0",
    "year": 2017,
    "contributor": "none",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# 这里是你数据集的类别，这里有三个分类，就是square, circle, triangle。制作自己的数据集主要改这里就行了
CATEGORIES = [
    {
        'id': 1,
        'name': 'square',
        'supercategory': 'shape',
    },
    {
        'id': 2,
        'name': 'circle',
        'supercategory': 'shape',
    },
    {
        'id': 3,
        'name': 'triangle',
        'supercategory': 'shape',
    },
]




coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}



image_id = 1
segmentation_id = 1

# 按照原始的Image，查找到对应的png，然后转换存入
ROOT_DIR = 'F:\Dataset\Dataset'  # 根目录
IMAGE_DIR = os.path.join(ROOT_DIR, "image")  # 根目录下存放你原图的文件夹
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")  # 根目录下存放mask标签的文件夹

for file_dir, label_dir in zip(os.listdir(IMAGE_DIR), os.listdir(ANNOTATION_DIR)):
    Sec_dirs = [os.path.join(os.path.join(IMAGE_DIR,file_dir), dir) for dir in os.listdir(os.path.join(IMAGE_DIR,file_dir))]
    for sec_dir in Sec_dirs:
        img_path_ls  = [os.path.join(sec_dir, path) for path in os.listdir(sec_dir)]
        for img_path in tqdm(img_path_ls):
            raw_path = img_path
            mask_dir = os.path.dirname(raw_path.replace(file_dir, label_dir).replace('image', 'annotations'))
            path_basename = os.path.basename(img_path)
            mask_path = glob(os.path.join(mask_dir, f"{path_basename.split('.')[0]}*"))[0]
            tool_id = os.path.basename(mask_path).split('.')[0][-2:]
            tool_id = tool_id[1:] if tool_id[0] == "_" else tool_id
            """"""

            image = Image.open(img_path)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(img_path), image.size)
            coco_output["images"].append(image_info)

            category_info = {'id': tool_id, 'is_crowd': 'crowd' in path_basename}
            im = Image.open(mask_path).convert('1')
            binary_mask = np.array(im).astype(np.uint8)
            annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
                segmentation_id = segmentation_id + 1

with open('{}/instances_shape_train2018.json'.format(ROOT_DIR), 'w') as output_json_file:
    json.dump(coco_output, output_json_file)