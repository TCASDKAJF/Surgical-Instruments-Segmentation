import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

def plot(image1, image2, image3, image4):
    """plot

    Args:
        image (numpy): load from dataloader with transform_raw
        tf_image (numpy): load from dataloader with transform1
        label (numpy): load from dataloader with transform2
        pred  (numpy): model prediction after argmax
    """

    fig, axs = plt.subplots(1, 4)

    image1 = np.transpose(image1, [1, 2, 0])
    axs[0].imshow(image1)

    image2 = np.transpose(image2, [1, 2, 0])
    axs[1].imshow(image2)

    image3 = np.transpose(image3, [1, 2, 0])
    axs[2].imshow(image3)

    image4 = np.transpose(image4, [1, 2, 0])
    axs[3].imshow(image4)
    plt.savefig('res.png')

def metrics(pred, mask):
    correct  = torch.eq(pred, mask).int()   
    accuracy = float(correct.sum()) / float(correct.numel()) 
    return accuracy

def MIoU(pred, mask):
    ious = []
    for cls in range(3):
        pred_inds = pred == cls
        target_inds = mask == cls
        intersection = (pred_inds[target_inds]).sum().float()
        union = pred_inds.sum().float() + target_inds.sum().float() - intersection
        iou = intersection / union
        ious.append(iou)
    
    miou = sum(ious) / len(ious)
    return miou.detach().cpu().numpy()


def transform_mask(mask):
    mask_prob = np.zeros((len(mask), 2, 256, 256), dtype=np.uint8)
    mask_np = mask.detach().numpy().astype(np.uint8)
    for i in range(len(mask)):
        mask_prob[i][0][mask[i] != 0] = 1
    mask_trans = torch.tensor(mask_prob)
    return mask_trans

