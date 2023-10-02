import os
import datetime

import torch
from torchvision.ops.misc import FrozenBatchNorm2d
from backbone_new.backbone import swin_fpn_backbone
from network_files import MaskRCNN
from models import resnet50_fpn_backbone
from Dataset.Dataset_coco import CocoDetection
from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from models.Optimizer import Lion
from tqdm import tqdm, trange
from scipy.optimize import differential_evolution
from torchvision.transforms import ColorJitter, GaussianBlur, RandomAffine, RandomHorizontalFlip, RandomCrop
from torchvision.models.detection.rpn import AnchorGenerator


stored_mAP = 0.0

def create_model(num_classes, load_pretrain_weights=True, anchor_sizes=None):
    # Use the default anchor size, unless another value is provided
    initial_anchors = [26, 62, 134, 286, 626]
    if anchor_sizes is None:
        anchor_sizes = tuple((size,) for size in initial_anchors)

    # create an anchor generator for the FPN
    rpn_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    )

    backbone = swin_fpn_backbone("swin_tiny","/root/autodl-tmp/new_swin/Detection_Baseline_swin/Weight/swin_tiny.pth")
    # backbone = resnet50_fpn_backbone(pretrain_path="Weight/resnet50.pth", trainable_layers=3)

    # Passing the new anchor generator to the MaskRCNN model
    model = MaskRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=rpn_anchor_generator)
    # model = MaskRCNN(backbone, num_classes=num_classes)
    
#     if load_pretrain_weights:
#         # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
#         weights_dict = torch.load("Weight/maskrcnn_resnet50_fpn.pth", map_location="cpu")
#         for k in list(weights_dict.keys()):
#             if ("box_predictor" in k) or ("mask_fcn_logits" in k):
#                 del weights_dict[k]

#         print(model.load_state_dict(weights_dict, strict=False))

    return model



def get_mAP_loss(*args):
    global stored_mAP  # claim global variable
    return -stored_mAP


def evaluate_with_anchors(anchor_sizes, train_data_loader, val_data_loader, model, optimizer, device, scaler):

    # use the given anchor sizes to update the model
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain, anchor_sizes=anchor_sizes)
    model.to(device)

    # train the model for one epoch
    utils.train_one_epoch(model, optimizer, train_data_loader, device, 0, print_freq=50, warmup=True, scaler=scaler)

    # validate the model
    det_info, _ = utils.evaluate(model, val_data_loader, device=device)

    # Return negative mAP as a loss (since we want to maximize mAP)
    return -det_info[1]


def update_model_anchors(model, new_anchor_sizes):
    """
    Update the anchor sizes of the given model.

    Parameters:
    - model: The current MaskRCNN model.
    - new_anchor_sizes: A list of new anchor sizes.

    Returns:
    - model: The MaskRCNN model with updated anchor sizes.
    """

    # Convert the new anchor dimensions to the appropriate format
    anchor_sizes = tuple((size,) for size in new_anchor_sizes)

    # Create a new anchor generator
    rpn_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    )

    # Updating the model's anchor generator
    model.rpn.anchor_generator = rpn_anchor_generator

    return model


def main(args):
    global stored_mAP  # claim global variable
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # save coco_info
    results_dir = "cbam_2/"  # Specify a directory
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"{results_dir}det_results_swin{now}.txt"
    seg_results_file = f"{results_dir}seg_results_swin{now}.txt"

    data_root = args.data_path


    # load train data set
    # coco -> annotations -> instances_train.json
    train_dataset = CocoDetection(data_root, "train", mode='train')
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    # train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt", transforms=data_transform["train"])
    train_sampler = None

    # Whether to compose a batch by sampling images with similar aspect ratios.
    # This reduces the amount of GPU memory required for training, and is used by default.
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 8
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # coco -> annotations -> instance_val.json
    val_dataset = CocoDetection(data_root, "val", mode='val')
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model.to(device)

    train_loss = []
    learning_rate = []
    val_map = []

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optimizer = Lion(params, lr=0.0004, betas=(0.0, 0.99))

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)
    # if resume training, load the last weights
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        path = args.output_dir + '/' +f"model_{args.start_epoch}.pth"

        print(path)
        checkpoint = torch.load(path, map_location='cpu')  # load checkpoint
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

        # write detection into txt
        with open(det_results_file, "a") as f:
            # write the data including coco index, loss and learning rate
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # write seg into txt
        with open(seg_results_file, "a") as f:
            # write the data including coco index, loss and learning rate
            result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(det_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "./save_weights/cbam_2/model_{}.pth".format(epoch))
        
        
#         # At the end of each epoch, the optimal anchor size is re-evaluated using a differential evolutionary algorithm
        if epoch % args.reevaluate_anchors_every == 0:  
            print("Re-evaluating anchor sizes...")
            initial_anchors = [26, 62, 134, 286, 626]
            # Set the bounds for the anchor sizes
            bounds = [(anchor-10, anchor+10) for anchor in initial_anchors]
            stored_mAP = det_info[1]
            result = differential_evolution(get_mAP_loss, bounds)
            best_anchor_sizes = result.x
            print(f"Best anchor sizes found: {best_anchor_sizes}")

            # Update the model with the new anchor sizes
            model = update_model_anchors(model, best_anchor_sizes)
            model.to(device)
            

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # Device
    parser.add_argument('--device', default='cuda:0', help='device')
    # The root directory of the training dataset
    parser.add_argument('--data-path', default='/root/autodl-tmp/new_swin/Detection_Baseline_swin/data_new/coco', help='dataset')
    # Number of detection target categories (without background)
    parser.add_argument('--num-classes', default=16, type=int, help='num_classes')
    # File save address
    parser.add_argument('--output-dir', default='save_weights', help='path where to save')
    # If you need to follow the last training, specify the address of the last training weights file.
    parser.add_argument('--resume', default=False, type=str, help='resume from checkpoint')
    # Specify the number of epochs to start training from.
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # Total number of epochs to train for
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    # Learning rate
    parser.add_argument('--lr', default=0.0041, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    
    # Update your anchor size every however many epochs!
    parser.add_argument('--reevaluate_anchors_every', type=int, default=3, 
                    help='Number of epochs after which to re-evaluate and adjust anchor sizes using differential evolution.')

    # SGD momentum
    parser.add_argument('--momentum', default=0.901, type=float, metavar='M',
                        help='momentum')
    # SGD Weight decay
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # Arguments against torch.optim.lr_scheduler.MultiStepLR
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # Arguments against torch.optim.lr_scheduler.MultiStepLR
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # Batch size for training (larger settings are recommended if memory/GPU graphics are plentiful)
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=True, help="load COCO pretrain weights.")
    # Whether to train with mixed precision (requires GPU support for mixed pre
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # Check if the save weights folder exists, create it if it doesn't
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)


    """
    nohup python train.py > train.log 2>&1 &
    1046779
    """