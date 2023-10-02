import os
import datetime
import wandb
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
from argparse import Namespace


os.environ["WANDB_API_KEY"] = "881be73088d2f7520f6cf037426acde0fb775d4b"
wandb.login()


stored_mAP = 0.0

def create_model(num_classes, load_pretrain_weights=True, anchor_sizes=None):
    # 使用默认的锚点尺寸，除非提供了其他值
    initial_anchors = [26, 62, 134, 286, 626]
    if anchor_sizes is None:
        anchor_sizes = tuple((size,) for size in initial_anchors)

    # 创建一个新的锚点生成器
    rpn_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    )

    backbone = swin_fpn_backbone("swin_tiny","/root/autodl-tmp/new_swin/Detection_Baseline_swin/Weight/swin_tiny.pth")
    # backbone = resnet50_fpn_backbone(pretrain_path="Weight/resnet50.pth", trainable_layers=3)

    # 将新的锚点生成器传递给MaskRCNN模型
    # model = MaskRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=rpn_anchor_generator)
    model = MaskRCNN(backbone, num_classes=num_classes)
    
#     if load_pretrain_weights:
#         # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
#         weights_dict = torch.load("Weight/maskrcnn_resnet50_fpn.pth", map_location="cpu")
#         for k in list(weights_dict.keys()):
#             if ("box_predictor" in k) or ("mask_fcn_logits" in k):
#                 del weights_dict[k]

#         print(model.load_state_dict(weights_dict, strict=False))

    return model



def get_mAP_loss(*args):
    global stored_mAP  # 声明为全局变量
    return -stored_mAP


def evaluate_with_anchors(anchor_sizes, train_data_loader, val_data_loader, model, optimizer, device, scaler):

    # 使用给定的锚点尺寸创建模型
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain, anchor_sizes=anchor_sizes)
    model.to(device)

    # 训练模型（这里只进行一个epoch，但你可以根据需要进行更多的epoch）
    utils.train_one_epoch(model, optimizer, train_data_loader, device, 0, print_freq=50, warmup=True, scaler=scaler)

    # 在验证集上评估模型
    det_info, _ = utils.evaluate(model, val_data_loader, device=device)

    # 返回负的mAP作为损失（因为我们希望最大化mAP）
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

    # 将新的锚点尺寸转换为适当的格式
    anchor_sizes = tuple((size,) for size in new_anchor_sizes)

    # 创建一个新的锚点生成器
    rpn_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    )

    # 更新模型的锚点生成器
    model.rpn.anchor_generator = rpn_anchor_generator

    return model


def main(args, config):
    global stored_mAP  # 声明为全局变量
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"

    data_root = args.data_path

    # load train data set
    # coco -> annotations -> instances_train.json
    train_dataset = CocoDetection(data_root, "train", mode='train')
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    # train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt", transforms=data_transform["train"])
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
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
    
    # ======================================================================
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=config.project_name, config=config.__dict__, name=nowtime, save_code=True)
    print("wandb.run.id = ", wandb.run.id)
    # ======================================================================

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
    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        path = args.output_dir + '/' +f"model_{args.start_epoch}.pth"

        print(path)
        checkpoint = torch.load(path, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
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
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # write seg into txt
        with open(seg_results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
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
        torch.save(save_files, "./save_weights/model_{}.pth".format(epoch))
        
        wandb.log({'epoch': epoch, 'loss': mean_loss})
        
        
#         # 在每个epoch结束后，使用差分进化算法重新评估最佳的锚点尺寸
#         if epoch % args.reevaluate_anchors_every == 0:  
#             print("Re-evaluating anchor sizes...")
#             initial_anchors = [26, 62, 134, 286, 626]
#             # 这里表示在你文件中给出的[26, 62, 134, 286, 626]的上下10浮动，如果想要浮动大一些可以修改
#             bounds = [(anchor-10, anchor+10) for anchor in initial_anchors]
#             stored_mAP = det_info[1]
#             result = differential_evolution(get_mAP_loss, bounds)
#             best_anchor_sizes = result.x
#             print(f"Best anchor sizes found: {best_anchor_sizes}")

#             # 更新模型的锚点尺寸
#             model = update_model_anchors(model, best_anchor_sizes)
#             model.to(device)
            
    wandb.finish()
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

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='/root/autodl-tmp/new_swin/Detection_Baseline_swin/data_new/coco', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=16, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default=False, type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.004, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    
    # 每多少个epoch更新一次你的锚点尺寸
    parser.add_argument('--reevaluate_anchors_every', type=int, default=3, 
                    help='Number of epochs after which to re-evaluate and adjust anchor sizes using differential evolution.')

    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练的batch size(如果内存/GPU显存充裕，建议设置更大)
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=True, help="load COCO pretrain weights.")
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    configure = Namespace(

    project_name='MaskRCNN',
    num_classes=args.num_classes,
    batch_size=args.batch_size,
    lr=args.lr,
    epochs=args.epochs,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    lr_steps=args.lr_steps,
    lr_gamma=args.lr_gamma,

    )

    main(args, configure)


    """
    nohup python train.py > train.log 2>&1 &
    1046779
    """