import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
# from backbone_old.resnet50_fpn_model import resnet50_fpn_backbone
from backbone_new.backbone import swin_fpn_backbone
from draw_box_utils import draw_objs


# def create_model(num_classes, box_thresh=0.5):
#     # create model
#     backbone = resnet50_fpn_backbone(pretrain_path="Weight/resnet50.pth", trainable_layers=3)
#     model = MaskRCNN(backbone, num_classes=args.num_classes + 1)
    
#     backbone = resnet50_fpn_backbone()
#     model = MaskRCNN(backbone,
#                      num_classes=num_classes,
#                      rpn_score_thresh=box_thresh,
#                      box_score_thresh=box_thresh)

#     return model


# def time_synchronized():
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     return time.time()


# def main():
#     num_classes = 16  # 不包含背景
#     box_thresh = 0.5
#     weights_path = "/root/autodl-tmp/Detection_Baseline_swin/work_dir/maskrcnn_data_augmentation/model_24.pth"
#     img_path = "/root/autodl-tmp/Detection_Baseline_swin/data/video01_clip01_step1_023.jpg"
#     label_json_path = './coco91_indices.json'

#     # get devices
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("using {} device.".format(device))

#     # create model
#     model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

#     # load train weights
#     assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
#     weights_dict = torch.load(weights_path, map_location='cpu')
#     weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
#     model.load_state_dict(weights_dict)
#     model.to(device)

#     # read class_indict
#     assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
#     with open(label_json_path, 'r') as json_file:
#         category_index = json.load(json_file)

#     # load image
#     assert os.path.exists(img_path), f"{img_path} does not exits."
#     original_img = Image.open(img_path).convert('RGB')

#     # from pil image to tensor, do not normalize image
#     data_transform = transforms.Compose([transforms.ToTensor()])
#     img = data_transform(original_img)
#     # expand batch dimension
#     img = torch.unsqueeze(img, dim=0)

#     model.eval()  # 进入验证模式
#     with torch.no_grad():
#         # init
#         img_height, img_width = img.shape[-2:]
#         init_img = torch.zeros((1, 3, img_height, img_width), device=device)
#         model(init_img)

#         t_start = time_synchronized()
#         predictions = model(img.to(device))[0]
#         t_end = time_synchronized()
#         print("inference+NMS time: {}".format(t_end - t_start))

#         predict_boxes = predictions["boxes"].to("cpu").numpy()
#         predict_classes = predictions["labels"].to("cpu").numpy()
#         predict_scores = predictions["scores"].to("cpu").numpy()
#         predict_mask = predictions["masks"].to("cpu").numpy()
#         predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

#         if len(predict_boxes) == 0:
#             print("没有检测到任何目标!")
#             return

#         plot_img = draw_objs(original_img,
#                              boxes=predict_boxes,
#                              classes=predict_classes,
#                              scores=predict_scores,
#                              masks=predict_mask,
#                              category_index=category_index,
#                              line_thickness=3,
#                              font='arial.ttf',
#                              font_size=20)
#         plt.imshow(plot_img)
#         plt.show()
#         # 保存预测的图片结果
#         plot_img.save("test_result.jpg")


# if __name__ == '__main__':
#     main()


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def process_image(img_path, model, category_index, device):
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        predictions = model(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()
        predict_mask = np.squeeze(predict_mask, axis=1)

        if len(predict_boxes) == 0:
            print(f"No objects detected in {img_path}")
            return original_img

        plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        return plot_img

def main(args):
    
    num_classes = 16
    box_thresh = 0.5
    weights_path ='/root/autodl-tmp/new_swin/Detection_Baseline_swin/save_weights/model_29.pth'
    imgs_dir = "/root/autodl-tmp/new_swin/Detection_Baseline_swin/data_new/coco/test"
    label_json_path = './coco91_indices.json'
    results_dir = os.path.join(imgs_dir, "results_threshold_0.4")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    backbone = swin_fpn_backbone("swin_tiny","/root/autodl-tmp/new_swin/Detection_Baseline_swin/Weight/swin_tiny.pth")
    model = MaskRCNN(backbone, num_classes=args.num_classes + 1)
    # backbone = resnet50_fpn_backbone(pretrain_path="Weight/resnet50.pth", trainable_layers=3)
    # model = MaskRCNN(backbone, num_classes=args.num_classes + 1)

    # 载入你自己训练好的模型权重
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    img_files = [f for f in os.listdir(imgs_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    os.makedirs(results_dir, exist_ok=True)

    for img_file in img_files:
        img_path = os.path.join(imgs_dir, img_file)
        processed_img = process_image(img_path, model, category_index, device)
        save_path = os.path.join(results_dir, img_file)
        processed_img.save(save_path)

if __name__ == '__main__':
    
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

    main(args)