import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Load up the the finetuned model
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# sam_finetuned = sam_model_registry['vit_b'](checkpoint='/sam-finetuning/epoch-2-loss-81.46-ckpt.pth')
# sam_finetuned.to(device)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def main():
    # sam = sam_model_registry["vit_b"](
    #     checkpoint='/data/result/changxiu/open_source/SAM/model/sam_vit_b_01ec64.pth')
    sam = sam_model_registry["vit_b"](checkpoint='/data/result/changxiu/open_source/SAM/coco_seg/sam_finetuning_Dice_batch1/epoch-2-trainloss-0.94-valloss-0.95-ckpt.pth')
    # sam = sam_model_registry["vit_h"](checkpoint="./model/sam_vit_h_4b8939.pth")
    sam.to(device)
    # predictor = SamPredictor(sam)

    mask_generator = SamAutomaticMaskGenerator(sam)


    base_path = '/data/result/changxiu/open_source/SAM/coco_seg_dataset/test2017/'
    f_list = os.listdir(base_path)

    save_base = '/data/result/changxiu/open_source/SAM/finetune_result/test_imgs/'
    os.makedirs(save_base, exist_ok=True)

    for f_name in f_list:
        # if f_name != '2.jpg':
        #     continue
        print('process:', f_name)
        img = cv2.imread(base_path + f_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # predictor.set_image(img)
        # masks, _, _ = predictor.predict()

        masks = mask_generator.generate(img)
        # print(img.shape, masks.shape, np.unique(masks))
        print(len(masks))

        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        show_anns(masks)
        plt.axis('off')
        plt.savefig(save_base + f_name)
        plt.show()

        # cv2.imwrite(save_base + f_name, 255*masks.transpose([1, 2, 0]))



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


if __name__ == "__main__":
    main()
