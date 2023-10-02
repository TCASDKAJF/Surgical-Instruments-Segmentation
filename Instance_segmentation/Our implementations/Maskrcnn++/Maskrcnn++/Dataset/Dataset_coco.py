import os
import json

import matplotlib.pyplot as plt
import torch.utils.data as data
from pycocotools.coco import COCO
from train_utils import coco_remove_images_without_annotations, convert_coco_poly_mask
import torch
import random
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.transforms import functional as F
import math
import numpy as np


class CocoDetection(data.Dataset):
    dict_amount = {
        'bipolar_forceps': 116,
        'blakesley': 86,
        'cottle': 255,
        'cup_forceps': 251,
        'doppler': 66,
        'drill': 85,
        'dural_scissors': 494,
        'freer_elevator': 1148,
        'kerrisons': 4537,
        'pituitary_rongeurs': 1378,
        'retractable_knife': 1134,
        'ring_curette': 154,
        'spatula_dissector': 597,
        'stealth_pointer': 284,
        'suction': 4868,
        'surgiflo': 304
    }

    def __init__(self, root, dataset="train", img_resize=1024, xyxy=True, mode='train'):
        super(CocoDetection, self).__init__()
        assert dataset in ["train", "val", "test"], 'dataset must be in ["train", "val"，"test"]'
        anno_file = f"instance_{dataset}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, f"{dataset}")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)
        self.mode = mode
        self.coco = COCO(self.anno_path)
        self.image_size = img_resize

        # 获取coco数据索引与类别名称的关系
        # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        max_index = max(data_classes.keys())  # 90
        # 将缺失的类别名称设置成N/A
        coco_classes = {}
        for k in range(0, max_index + 1):
            if k in data_classes:
                coco_classes[k] = data_classes[k]
            else:
                coco_classes[k] = "N/A"

        if dataset == "train":
            json_str = json.dumps(coco_classes, indent=4)
            with open("coco91_indices.json", "w") as f:
                f.write(json_str)

        self.coco_classes = coco_classes
        self.xyxy = xyxy
        self.to_tensor = ToTensor()

        ids = list(sorted(self.coco.imgs.keys()))
        if dataset == "train":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = coco_remove_images_without_annotations(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # 1,4
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_mask(segmentations, h, w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        new_bboxes = []
        for bbox in boxes:
            bbox = self.xyxy_to_xywh(bbox)  # xyxy_to_xywh , 进行增强
            new_bboxes.append(bbox)
        boxes = torch.tensor(new_bboxes)

        target = {"boxes": boxes,  # xywh
                  "labels": classes,
                  "masks": masks,
                  "image_id": torch.tensor([img_id]),
                  "area": area,
                  "iscrowd": iscrowd}
        return target

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')
        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h)
        """
        img: tensor
        target: dict
        """
        if self.mode == 'train':
            img = self.to_tensor(img)
            img, mask, boxes = self.data_enhance(img, target['masks'], target['boxes'], target['labels'])
        else:
            mask, boxes = target['masks'], target['boxes']
            img = self.to_tensor(img)

        # torch.Size([3, 720, 1280]) torch.Size([1, 720, 1280]) torch.Size([1, 4])
        if self.xyxy:
            new_bboxes = []
            for bbox in boxes:
                bbox = self.xywh_to_xyxy(bbox)  # 翻转对应bbox坐标信息
                new_bboxes.append(bbox)
            boxes = torch.tensor(new_bboxes)

        # torch.Size([3, 720, 1280]) torch.Size([1, 720, 1280]) torch.Size([1, 4])

        target['masks'] = mask
        target['boxes'] = boxes
        # img = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
        return img, target

    def __len__(self):
        return len(self.ids)

    def data_enhance(self, image, mask, bboxes, target_labels):
        # ([3, 720, 1280]) torch.Size([1, 720, 1280]) torch.Size([1, 1, 4])
        height, width = image.shape[-2:]

        # 获取每一个label的对应增强概率，保持样本数量均衡
        min_value = 4868
        for label in target_labels:
            if int(label) == 0:
                continue
            class_name = self.coco_classes[int(label)]
            amount = self.dict_amount[class_name]
            if amount < min_value:
                min_value = amount

        enhance_ratio = 15757 / min_value
        enhance_ratio = enhance_ratio / 185  # norm [0-1]


        # 彩色图数据增强
        image = F.adjust_gamma(image, gamma=random.uniform(0.8, 1.2))
        image = F.adjust_contrast(
            image, contrast_factor=random.uniform(0.8, 1.2))
        image = F.adjust_brightness(
            image, brightness_factor=random.uniform(0.8, 1.2))
        image = F.adjust_saturation(
            image, saturation_factor=random.uniform(0.8, 1.2))
        image = F.adjust_hue(image, hue_factor=random.uniform(-0.2, 0.2))
        image = F.adjust_sharpness(
            image, sharpness_factor=random.uniform(0.8, 1.2))
        # 同步op
        image_mask = torch.cat([image, mask], dim=0)

        # if self.image_size:
        #     image_mask = F.resize(image_mask, size=[self.image_size, self.image_size])
        #     new_bboxes = []
        #     for bbox in bboxes:
        #         bbox = self.resize_bbox(bbox, width, height, self.image_size, self.image_size)  # 翻转对应bbox坐标信息
        #         new_bboxes.append(bbox)
        #     bboxes = torch.tensor(new_bboxes)

        if random.uniform(0, 1) > 0.5:
            image_mask = F.hflip(image_mask)
            new_bboxes = []
            for bbox in bboxes:
                bbox = self.flip_bbox(bbox, width, height, mode=1)  # 修改对应bbox坐标信息
                new_bboxes.append(bbox)
            bboxes = torch.tensor(new_bboxes)
        if random.uniform(0, 1) > enhance_ratio/2:
            image_mask = F.vflip(image_mask)
            new_bboxes = []
            for bbox in bboxes:
                bbox = self.flip_bbox(bbox, width, height, mode=0)  # 修改对应bbox坐标信息
                new_bboxes.append(bbox)
            bboxes = torch.tensor(new_bboxes)
        if random.uniform(0, 1) > enhance_ratio/2:
            angle = random.uniform(-20, 20)
            image_mask = F.rotate(image_mask, angle=angle)

            new_bboxes = []
            for bbox in bboxes:
                bbox = self.rotate_bbox(bbox, width, height, angle)  # 修改对应bbox坐标信息
                new_bboxes.append(bbox)
            bboxes = torch.tensor(new_bboxes)

        # ([3, 720, 1280]) torch.Size([1, 720, 1280])
        image = image_mask[:3, ...]
        mask = image_mask[3:, ...]
        return image, mask, bboxes

    def resize_bbox(self, bbox, img_width, img_height, new_width, new_height):
        # bbox格式为[x_min, y_min, x_max, y_max]
        # 返回缩放后的bbox
        return [bbox[0] * new_width / img_width,
                bbox[1] * new_height / img_height,
                bbox[2] * new_width / img_width,
                bbox[3] * new_height / img_height]

    def flip_bbox(self, bbox, img_width, img_height, mode):
        # bbox格式为[x_min, y_min, width, height]
        # mode为翻转模式，0为垂直翻转，1为水平翻转
        # 返回翻转后的bbox
        if mode == 0:  # 垂直翻转
            return [bbox[0], img_height - bbox[1] - bbox[3], bbox[2], bbox[3]]
        elif mode == 1:  # 水平翻转
            return [img_width - bbox[0] - bbox[2], bbox[1], bbox[2], bbox[3]]
        else:
            return bbox

    def rotate_bbox(self, bbox, img_width, img_height, angle):
        # bbox格式为[x_min, y_min, width, height]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1]
        x3 = bbox[0] + bbox[2]
        y3 = bbox[1] + bbox[3]
        x4 = bbox[0]
        y4 = bbox[1] + bbox[3]
        # 计算旋转中心点坐标
        center_x = img_width // 2
        center_y = img_height // 2
        # 计算旋转角度的弧度值
        radian = math.radians(-angle)
        # 计算旋转后的四个顶点坐标
        x1_ = x1 * math.cos(radian) - y1 * math.sin(radian) + center_x - center_x * math.cos(
            radian) + center_y * math.sin(radian)
        y1_ = x1 * math.sin(radian) + y1 * math.cos(radian) + center_y - center_x * math.sin(
            radian) - center_y * math.cos(radian)
        x2_ = x2 * math.cos(radian) - y2 * math.sin(radian) + center_x - center_x * math.cos(
            radian) + center_y * math.sin(radian)
        y2_ = x2 * math.sin(radian) + y2 * math.cos(radian) + center_y - center_x * math.sin(
            radian) - center_y * math.cos(radian)
        x3_ = x3 * math.cos(radian) - y3 * math.sin(radian) + center_x - center_x * math.cos(
            radian) + center_y * math.sin(radian)
        y3_ = x3 * math.sin(radian) + y3 * math.cos(radian) + center_y - center_x * math.sin(
            radian) - center_y * math.cos(radian)
        x4_ = x4 * math.cos(radian) - y4 * math.sin(radian) + center_x - center_x * math.cos(
            radian) + center_y * math.sin(radian)
        y4_ = x4 * math.sin(radian) + y4 * math.cos(radian) + center_y - center_x * math.sin(
            radian) - center_y * math.cos(radian)
        # 计算旋转后的bbox的左上角坐标和宽高
        x_ = min(x1_, x2_, x3_, x4_)
        y_ = min(y1_, y2_, y3_, y4_)
        w_ = max(x1_, x2_, x3_, x4_) - min(x1_, x2_, x3_, x4_)
        h_ = max(y1_, y2_, y3_, y4_) - min(y1_, y2_, y3_, y4_)
        return [x_, y_, w_, h_]

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def xywh_to_xyxy(bbox):
        # bbox格式为[x_min, y_min, width, height]
        # 返回转换后的bbox
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    @staticmethod
    def xyxy_to_xywh(bbox):
        # bbox格式为[x_min, y_min, x_max, y_max]
        # 返回转换后的bbox
        return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    import cv2

    mydata = CocoDetection('root/autodl-tmp/Detection_Baseline_swin/data/coco/train', dataset="train")
    func = ToPILImage()
    my_load = data.DataLoader(mydata, batch_size=1, shuffle=True)
    for img, target in my_load:
        plt.subplot(2, 2, 1)
        image = np.array(func(img[0]))
        for anns in target['boxes'][0]:
            x1 = int(anns[0])
            y1 = int(anns[1])
            x2 = int(anns[2])
            y2 = int(anns[3])

            cv2.rectangle(image,
                          (x1, y1),
                          (x2, y2),
                          (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
        plt.imshow(image)

        for i, mask in enumerate(target['masks'][0]):
            print(mask.shape)
            mask = func(mask)
            plt.subplot(2, 2, i + 2)
            plt.imshow(mask)
        plt.show()
