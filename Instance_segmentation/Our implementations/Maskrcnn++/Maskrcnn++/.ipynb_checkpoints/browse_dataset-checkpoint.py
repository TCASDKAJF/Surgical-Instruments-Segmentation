import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from os.path import join as opj

#
# def coco_segmentation_to_mask(coco_annotation, image_id):
#     annotation = coco_annotation.loadAnns(coco_annotation.getAnnIds(imgIds=image_id))
#     img_info = coco_annotation.loadImgs(image_id)[0]
#     img_height = img_info['height']
#     img_width = img_info['width']
#
#     masks = []
#     ann_info = []
#     for ann in annotation:
#         if 'segmentation' in ann:
#             rle = maskUtils.frPyObjects(ann['segmentation'], img_height, img_width)
#             mask = maskUtils.decode(rle)
#             masks.append(mask)
#             ann_info.append([ann['bbox'], ann['category_id'], ann['image_id']])
#
#
#     final_mask = np.zeros((img_height, img_width,1), dtype=np.uint8)
#     for mask in masks:
#         final_mask += mask
#
#     return final_mask, ann_info
#
# from pycocotools.coco import COCO
#
# # 加载COCO注释文件
# coco_annotation = COCO(r'E:\acoding_task\mmdetection-main\coco_seg\annotations\instance_test.json')
#
# # 获取图像ID
# image_id = 1
#
# # 调用coco_segmentation_to_mask函数
# mask, info = coco_segmentation_to_mask(coco_annotation, image_id)
# plt.imshow(mask)
# plt.show()
# print(info)


def coco_segmentation_to_mask(coco_annotation, path_dir, image_id):
    # 原始图像加载
    img_info = coco_annotation.loadImgs(image_id)
    print(img_info)
    img_info = img_info[0]
    img_path = opj(path_dir,img_info['file_name'])

    img = cv2.imread(img_path)[:, :, ::-1]  # RGB

    # mask生成
    annotation = coco_annotation.loadAnns(coco_annotation.getAnnIds(imgIds=image_id))
    img_height = img_info['height']
    img_width = img_info['width']

    masks = []
    for ann in annotation:
        if 'segmentation' in ann:
            rle = maskUtils.frPyObjects(ann['segmentation'], img_height, img_width)
            mask = maskUtils.decode(rle)
            masks.append(mask)

    final_mask = np.zeros((img_height, img_width, 1), dtype=np.uint8)
    for mask in masks:
        final_mask += mask

    return img, final_mask


# COCO注释文件加载
coco_annotation = COCO(r'E:\acoding_task\mmdetection-main\coco_seg\annotations\instance_test.json')
# 调用函数获取图像和mask
image_id = 1
image, mask = coco_segmentation_to_mask(coco_annotation, r'E:\acoding_task\mmdetection-main\coco_seg\test', image_id)

print(image.shape, mask.shape)
# 在这里对原始图像和mask进行你需要的处理
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(mask)
plt.show()