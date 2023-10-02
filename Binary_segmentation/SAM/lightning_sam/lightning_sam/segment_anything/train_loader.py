import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class NDIS(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0][:1000]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0] # return all the dictionaries of image info with image_id
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id) # return annotation IDs with iamge IDs
        anns = self.coco.loadAnns(ann_ids) # return all the dictionaries of annotations with annotations IDs
        bboxes = []
        masks = []
        # print('img:', image_id, ann_ids, image_path, anns)

        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h]) # from bbox information, append x, y, x + w, y + h for SAM input
            mask = self.coco.annToMask(ann) # extrack mask information from ann
            masks.append(mask)

        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes)) # apply transform to image, masks, and bboxes

        # print('bboxes:', bboxes)
        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        # print('final box:', bboxes.shape, masks.shape, np.unique(masks))

        # ## for batch_size > 1;  we assume a image has up to 4 seg_masks at most
        # num = len(bboxes)
        # final_bboxes = np.zeros([4, 4], np.float32)
        # final_masks = np.zeros([4, masks.shape[1], masks.shape[2]], np.float32)
        # final_bboxes[:num, :] = bboxes
        # final_masks[:num, :, :] = masks
        # return image, torch.tensor(final_bboxes), torch.tensor(final_masks).float()

        return image, torch.tensor(bboxes), torch.tensor(masks).float()


class NDIS_Seg(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0][:2000]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0] # return all the dictionaries of image info with image_id
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # print('img:', image.shape)

        ann_ids = self.coco.getAnnIds(imgIds=image_id) # return annotation IDs with iamge IDs
        anns = self.coco.loadAnns(ann_ids) # return all the dictionaries of annotations with annotations IDs
        bboxes = []
        # masks = []
        # print('img:', image_id, ann_ids, image_path, anns)
        masks = np.zeros([image.shape[0], image.shape[1]], np.uint8)

        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h]) # from bbox information, append x, y, x + w, y + h for SAM input
            mask = self.coco.annToMask(ann) # extrack mask information from ann
            # masks.append(mask)
            masks += mask


        masks = np.where(masks > 0, 1, 0).astype(np.uint8)
        # masks = np.expand_dims(masks, -1)
        # if self.transform:
        #     image, masks = self.transform(image, masks, bboxes=None) # apply transform to image, masks, and bboxes
        # print(masks.shape)

        image = cv2.resize(image, [512, 512], cv2.INTER_LINEAR)
        # print(image.shape)
        image = np.transpose(image, [2, 0, 1])
        image = image / 255.0

        masks = cv2.resize(masks, [512, 512], cv2.INTER_NEAREST)

        masks = np.expand_dims(masks, 0)


        return torch.tensor(image).float(), torch.tensor(masks).float()



def collate_fn(batch):
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size) # resize image to target_size, 1024x1024
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes=None):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        if bboxes is not None:
            masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks] # apply transform.apply_image() to mask
        else:
            # print(masks.shape)
            masks = torch.tensor(self.transform.apply_image(masks))
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)  # apply padding to image and mask
        if bboxes is not None:
            masks = [transforms.Pad(padding)(mask) for mask in masks]
            # Adjust bounding boxes
            bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
            bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

            return image, masks, bboxes
        else:
            masks = transforms.Pad(padding)(masks)
            print(masks.size())

            return image, masks


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = NDIS(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
    val = NDIS(root_dir=cfg.dataset.val.root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      transform=transform)

    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    print('iter num:', len(train_dataloader), len(val_dataloader))
    return train_dataloader, val_dataloader


def load_seg_datasets(cfg, img_size):

    def collate_fn_seg(batch):
        images, masks = zip(*batch)
        images = torch.stack(images)
        return images, masks

    transform = ResizeAndPad(img_size)
    train = NDIS_Seg(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
    val = NDIS_Seg(root_dir=cfg.dataset.val.root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      transform=transform)

    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  )
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                )
    print('iter num:', len(train_dataloader), len(val_dataloader))
    return train_dataloader, val_dataloader




import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

