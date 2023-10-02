#!--*-- coding: utf- --*--
import os

import numpy as np
import pycocotools.mask as mask_utils
import json
import cv2

'''
批量读取json文件对应的mask掩膜

'''


json_path = "outputs/preds/"
out_path = "outputs/mask/"
if not os.path.exists(out_path):
    os.mkdir(out_path)
json_list = os.listdir(json_path)
for file in json_list:

    with open(json_path + file, "r", encoding="utf-8") as f:
        content = json.load(f)

    # print(content)

    labels = content['labels']
    for m,label in zip(content['masks'], labels):
      
        img = np.array(mask_utils.decode(m), dtype=np.float32)
        img = img * 255
        file_name = out_path + file[:-5] + '____' + '___'  + str(label) + '.jpg'
        ret = cv2.imwrite(file_name, img)
        if not ret:
            print("failed")
   
