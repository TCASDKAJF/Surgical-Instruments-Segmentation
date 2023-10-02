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

#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)
 
    # 获取模型输出的feature/score
    model.eval()
    features = model.features(img)
    output = model.classifier(features)
 
    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g
 
    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
 
    features.register_hook(extract)
    pred_class.backward() # 计算梯度
 
    grads = features_grad   # 获取梯度
 
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
 
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]
 
    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
 
    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
    
    

if __name__ == '__main__':
    
    num_classes = 16
    box_thresh = 0.5
    weights_path ='/root/autodl-tmp/new_swin/Detection_Baseline_swin/save_weights/model_29.pth'
    imgs_dir = "/root/autodl-tmp/new_swin/Detection_Baseline_swin/data_new/coco/test/video01_clip01_step1_002.jpg"
    save_path = '/root/autodl-tmp/new_swin/Detection_Baseline_swin/head_result.jpg'
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
    draw_CAM(model,imgs_dir,save_path,)
    # model.to(device)
    # model.eval()

    # with open(label_json_path, 'r') as json_file:
        # category_index = json.load(json_file)

    # img_files = [f for f in os.listdir(imgs_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # os.makedirs(results_dir, exist_ok=True)

    # for img_file in img_files:
    #     img_path = os.path.join(imgs_dir, img_file)
    #     processed_img = process_image(img_path, model, category_index, device)
    #     save_path = os.path.join(results_dir, img_file)
    #     processed_img.save(save_path)