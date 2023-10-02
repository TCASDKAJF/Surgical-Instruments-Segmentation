import statistics
import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from train_loader import FocalLoss, DiceLoss, load_datasets, load_seg_datasets
from train_model import Model
from model.unet import UNet_2D

from my_predict import show_anns

# import segmentation_models_pytorch as smp

import os

if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

from box import Box

config = {
    "num_devices": 1,
    "batch_size": 6,
    "num_workers": 0,
    "num_epochs": 100,
    "eval_interval": 2,
    "out_dir": "/data/result/changxiu/open_source/SAM/coco_seg/Unet_seg_5000data_512/",
    "opt": {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60, 86],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        # "checkpoint": "/data/result/changxiu/open_source/SAM/model/sam_vit_b_01ec64.pth",
        "checkpoint": "/data/result/changxiu/open_source/SAM/coco_seg/sam_finetuning_Dice_batch1/epoch-0-valloss-8.73-ckpt.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/data/result/changxiu/open_source/SAM/coco_seg_dataset/train2017",
            "annotation_file": "/data/result/changxiu/open_source/SAM/coco_seg_dataset/annotations/instances_train2017.json"
        },
        "val": {
            "root_dir": "/data/result/changxiu/open_source/SAM/coco_seg_dataset/val2017",
            "annotation_file": "/data/result/changxiu/open_source/SAM/coco_seg_dataset/annotations/instances_val2017.json"
        }
    }
}

cfg = Box(config)


def lr_lambda(step):
    if step < cfg.opt.warmup_steps:
        return step / cfg.opt.warmup_steps
    elif step < cfg.opt.steps[0]:
        return 1.0
    elif step < cfg.opt.steps[1]:
        return 1 / cfg.opt.decay_factor
    else:
        return 1 / (cfg.opt.decay_factor ** 2)

def plot_contour(img, masks, save_path):
    img = np.asarray(img, np.uint8).copy()
    prediction = masks.astype("uint8") * 255  # 图像转为uint8类型
    ret_p, thresh_p = cv2.threshold(prediction, 127, 255, 0)  # 灰度图二值化，返回的第一个参数是阈值，第二个参数是阈值化后的图像
    contours_p, hierarchy_p = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours_p, -1, (0, 255, 0), 2)  ## 红色的为模型预测结果

    plt.plot()
    plt.imshow(img, cmap='gray')
    # plt.title(f_name)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path)
    plt.close()



def train_sam(cfg, model, optimizer, scheduler, train_dataloader, val_dataloader):
    train_loss = []

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    # os.makedirs('/data/result/changxiu/open_source/SAM/finetune_result/check_mask2/', exist_ok=True)

    for epoch in range(cfg.num_epochs):

        for iter, data in enumerate(train_dataloader):
            images, gt_mask = data
            # print('data:', images.size(), gt_mask.size())

            # img = torch.squeeze(images).detach().numpy().transpose([1, 2, 0]).astype(np.uint8)
            # mask = torch.squeeze(gt_mask).detach().numpy()
            # print(img.shape, mask.shape, np.unique(mask))
            # # import pdb
            # # pdb.set_trace()
            #
            # plot_contour(img, mask, '/data/result/changxiu/open_source/SAM/finetune_result/check_mask2/'+ str(iter+1) + '.png')



            # load batch on GPU device
            images = images.to(device)
            gt_mask = gt_mask.to(device)

            pred_mask = model(images)  # feed-forward
            loss_focal = focal_loss(pred_mask, gt_mask)
            loss_dice = dice_loss(pred_mask, gt_mask)


            # loss_total = 20. * loss_focal + loss_dice + loss_iou
            # loss_total = loss_dice + loss_focal
            loss_total = loss_dice
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(loss_total.item())

            ## len(train_dataloader)
            print(
                f'-- Epoch:{epoch + 1},{iter + 1}/{7879},total:{loss_total.item():.4f}, focall_loss:{loss_focal.item():.4f}, Dice_loss:{loss_dice.item():.4f}',
                end='\r')

            # if iter % 200 == 0:
            #     print(f'-- Epoch: [{epoch + 1}] Iteration: {iter + 1}/{len(train_dataloader)} --')
            #     print(f'Focal Loss [{loss_focal.item():.4f}] [avg: {avg_focal:.4f}]')
            #     print(f'Dice Loss [{loss_dice.item():.4f}] [avg: {avg_dice:.4f}]')
            #     print(f'IoU Loss [{loss_iou.item():.4f}] [avg: {avg_iou:.4f}]')
            #     print(f'Total Loss [{loss_total.item():.4f}] [avg: {avg_total:.4f}] \n')

        train_mean_loss = statistics.mean(train_loss)
        print('\n')
        print(f'--train Epoch: [{epoch + 1}] Mean Total Loss: [{train_mean_loss:.4f}] --\n')

        validate(model, val_dataloader, epoch, train_mean_loss)


def validate(model, val_dataloader, epoch, train_mean_loss):
    model.eval()  # turn the model into evaluation mode

    val_loss_list = []

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()

    with torch.no_grad():  # turn off requires_grad
        for iter, data in enumerate(val_dataloader):

            images, gt_mask = data
            gt_mask = gt_mask.to(device)

            # load batch on GPU device
            images = images.to(device)

            pred_mask = model(images)  # feed-forward
            loss_focal = focal_loss(pred_mask, gt_mask)
            loss_dice = dice_loss(pred_mask, gt_mask)

            # loss_total = 20. * loss_focal + loss_dice + loss_iou
            # loss_total = loss_dice + loss_focal
            loss_total = loss_dice
            val_loss_list.append(loss_total.item())

            print(
                f'-- Epoch:{epoch + 1},{iter + 1}/{7879},total:{loss_total.item():.4f}, focal_loss:{loss_focal.item():.4f}, Dice_loss:{loss_dice.item():.4f}',
                end='\r')


        total_loss_mean = statistics.mean(val_loss_list)
        print('\n')
        print(f'Validation [{epoch+1}]: Total Loss: [{total_loss_mean:.4f}]')

        print(f"\nSaving checkpoint to {cfg.out_dir}")
        state_dict = model.state_dict()
        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch}-trainloss-{train_mean_loss:.2f}-valloss-{total_loss_mean:.2f}-ckpt.pth"))


def main():
    # load SAM checkpoint
    model = UNet_2D()

    model = model.to(device)
    # model_path = '/data/result/changxiu/open_source/SAM/coco_seg/Unet_seg/epoch-19-trainloss-0.99-valloss-1.03-ckpt.pth'
    # model.load_state_dict(torch.load(model_path))
    # print('model load success!')

    train_data, val_data = load_seg_datasets(cfg, 512)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)

    # In the paper, the authors used AdamW for training.
    # optimizer = torch.optim.AdamW(model.model.parameters(), lr=cfg.opt.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.opt.weight_decay, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_sam(cfg, model, optimizer, scheduler, train_data, val_data)


def predict():
    base_path = '/data/result/changxiu/open_source/SAM/coco_seg_dataset/test2017/'
    f_list = os.listdir(base_path)

    model_path = '/data/result/changxiu/open_source/SAM/coco_seg/Unet_seg_all_data_dice/epoch-60-trainloss-0.85-valloss-0.86-ckpt.pth'

    save_base = '/data/result/changxiu/open_source/SAM/finetune_result/unet_all_data_epoch60_train/'
    os.makedirs(save_base, exist_ok=True)

    model = UNet_2D()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    print('model load success!')

    for f_name in f_list:
        # if f_name != '2.jpg':
        #     continue
        print('process:', f_name)
        img_ori = cv2.imread(base_path + f_name)
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        # predictor.set_image(img)
        # masks, _, _ = predictor.predict()

        image = np.transpose(img, [2, 0, 1])
        image = np.expand_dims(image, 0)
        image = torch.tensor(image).float().to(device)

        pred_out = model(image)
        masks = torch.squeeze(pred_out).cpu().detach().numpy()
        masks = np.where(masks > 0.5, 1, 0)
        # print(img.shape, masks.shape, np.unique(masks))
        print(len(masks))

        prediction = masks.astype("uint8") * 255  # 图像转为uint8类型
        ret_p, thresh_p = cv2.threshold(prediction, 127, 255, 0)  # 灰度图二值化，返回的第一个参数是阈值，第二个参数是阈值化后的图像
        contours_p, hierarchy_p = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        overlap_img = cv2.drawContours(img, contours_p, -1, (0, 255, 0), 2)  ## 红色的为模型预测结果

        plt.plot()
        plt.imshow(overlap_img, cmap='gray')
        plt.title(f_name)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_base + f_name)
        plt.close()





if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
    # predict()