import statistics
import torch
import torch.nn.functional as F
import os
from train_loader import FocalLoss, DiceLoss, load_datasets
from train_model import Model
from segment_anything import sam_model_registry

# import segmentation_models_pytorch as smp

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

from box import Box

config = {
    "num_devices": 1,
    "batch_size": 1,
    "num_workers": 0,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "/data/result/changxiu/open_source/SAM/coco_seg/sam_finetuning_Dice_batch1_0808/",
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
        "checkpoint": "/data/result/changxiu/open_source/SAM/coco_seg/sam_finetuning_Dice_batch1/epoch-1-valloss-7.71-ckpt.pth",
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


def train_sam(cfg, model, optimizer, scheduler, train_dataloader, val_dataloader):
    train_loss = []

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()

    for epoch in range(cfg.num_epochs):

        for iter, data in enumerate(train_dataloader):
            images, bboxes, gt_masks = data

            # load batch on GPU device
            images = images.to(device)
            bboxes = torch.stack(bboxes, dim=0)
            # bboxes = bboxes.cuda()
            bboxes = bboxes.to(device)
            bboxes = list(bboxes)
            gt_masks = torch.stack(gt_masks, dim=0)
            # gt_masks = gt_masks.cuda()
            gt_masks = gt_masks.to(device)
            gt_masks = list(gt_masks)

            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, bboxes)  # feed-forward
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=device)
            loss_dice = torch.tensor(0., device=device)
            loss_iou = torch.tensor(0., device=device)

            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                # compute batch_iou of pred_mask and gt_mask
                pred_mask = (pred_mask >= 0.5).float()
                intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
                union = torch.sum(pred_mask, dim=(1, 2))
                epsilon = 1e-7
                batch_iou = (intersection / (union + epsilon)).unsqueeze(1)

                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            # loss_total = 20. * loss_focal + loss_dice + loss_iou
            loss_total = loss_dice + loss_iou
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            avg_focal = loss_focal.item() / batch_size  # compute average loss of a batch
            avg_dice = loss_dice.item() / batch_size
            avg_iou = loss_iou.item() / batch_size
            avg_total = loss_total.item() / batch_size

            train_loss.append(loss_total.item())

            print(
                f'-- Epoch:{epoch + 1},{iter + 1}/{len(train_dataloader)},total:{avg_total:.4f}, foacl_loss:{avg_focal:.4f}, Dice_loss:{avg_dice:.4f}, IOU_loss:{avg_iou:.4f}',
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

            images, bboxes, gt_masks = data

            # load batch on GPU device
            images = images.to(device)
            bboxes = torch.stack(bboxes, dim=0)
            # bboxes = bboxes.cuda()
            bboxes = bboxes.to(device)
            bboxes = list(bboxes)
            gt_masks = torch.stack(gt_masks, dim=0)
            # gt_masks = gt_masks.cuda()
            gt_masks = gt_masks.to(device)
            gt_masks = list(gt_masks)

            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, bboxes)
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=device)
            loss_dice = torch.tensor(0., device=device)
            loss_iou = torch.tensor(0., device=device)

            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                # compute batch_iou of pred_mask and gt_mask
                pred_mask = (pred_mask >= 0.5).float()
                intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
                union = torch.sum(pred_mask, dim=(1, 2))
                epsilon = 1e-7
                batch_iou = (intersection / (union + epsilon)).unsqueeze(1)

                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            # loss_total = 20. * loss_focal + loss_dice + loss_iou
            loss_total = loss_dice + loss_iou
            val_loss_list.append(loss_total.item())

            avg_focal = loss_focal.item() / batch_size  # compute average loss of a batch
            avg_dice = loss_dice.item() / batch_size
            avg_iou = loss_iou.item() / batch_size
            avg_total = loss_total.item() / batch_size

            print(
                f'-- Epoch:{epoch + 1},{iter + 1}/{len(val_dataloader)},total:{avg_total:.4f}, foacl_loss:{avg_focal:.4f}, Dice_loss:{avg_dice:.4f}, IOU_loss:{avg_iou:.4f}',
                end='\r')

            # if iter % 100 == 0:
            #     print(f'-- Epoch: [{epoch}] Iteration: {iter + 1}/{len(val_dataloader)} --')
            #     print(f'Focal Loss [{loss_focal.item():.4f}] [avg: {avg_focal:.4f}]')
            #     print(f'Dice Loss [{loss_dice.item():.4f}] [avg: {avg_dice:.4f}]')
            #     print(f'IoU Loss [{loss_iou.item():8f}] [avg: {avg_iou:.8f}]')
            #     print(f'Total Loss [{loss_total.item():.4f}] [avg: {avg_total:.4f}] \n')

        total_loss_mean = statistics.mean(val_loss_list)
        print('\n')
        print(f'Validation [{epoch+1}]: Total Loss: [{total_loss_mean:.4f}]')

        print(f"\nSaving checkpoint to {cfg.out_dir}")
        state_dict = model.model.state_dict()
        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch}-trainloss-{train_mean_loss:.2f}-valloss-{total_loss_mean:.2f}-ckpt.pth"))


def main():
    # load SAM checkpoint
    sam = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint)

    model = Model(cfg, sam).to(device)
    model.setup()

    print('img_size:', model.model.image_encoder.img_size)

    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)

    # In the paper, the authors used AdamW for training.
    # optimizer = torch.optim.AdamW(model.model.parameters(), lr=cfg.opt.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.opt.weight_decay, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_sam(cfg, model, optimizer, scheduler, train_data, val_data)


if __name__ == "__main__":
    main()
