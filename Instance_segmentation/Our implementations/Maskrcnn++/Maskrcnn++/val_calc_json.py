"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json
import itertools
import datetime
import torch
from tqdm import tqdm
import numpy as np

import transforms
from backbone_new.backbone import swin_fpn_backbone
from backbone_old.resnet50_fpn_model import resnet50_fpn_backbone
from network_files import MaskRCNN
from my_dataset_coco import CocoDetection
from my_dataset_voc import VOCInstances
from train_utils import EvalCOCOMetric
from train_utils import train_eval_utils as utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import Normalize


# def compute_mIoU(gt_masks, pred_masks, num_classes):
#     per_class_iou = []
#     for i in range(num_classes):
#         temp_gt = np.array(gt_masks) == i
#         temp_pred = np.array(pred_masks) == i
#         intersection = np.logical_and(temp_gt, temp_pred).sum()
#         union = np.logical_or(temp_gt, temp_pred).sum()
#         if union == 0:
#             iou = 0
#         else:
#             iou = intersection / union
#         per_class_iou.append(iou)
#     mIoU = np.mean(per_class_iou)
#     return mIoU


def delete_all_files(path):
    """删除指定路径下的所有文件"""
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print("Error deleting file: {file_path}")
            print(e)
            

def compute_confusion_matrix(true, pred, num_classes):
    """
    Compute the confusion matrix from true and predicted labels.
    """
    cm = confusion_matrix(true, pred, labels=np.arange(num_classes))
    return cm


def plot_confusion_matrix(cm, class_names, cmap=plt.cm.Blues):
    """
    该函数打印并绘制混淆矩阵。
    """
    cm_normalized = normalize_confusion_matrix(cm) * 100  # 转换为百分比
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # 显示所有刻度，并设置标签
    ax.set(xticks=np.arange(cm_normalized.shape[1]),
           yticks=np.arange(cm_normalized.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title="Confusion Matrix",
           ylabel="True label",
           xlabel="Predicted label")

    # 旋转标签并设置对齐方式
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 循环遍历数据并在相应的位置添加文本
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, format(int(cm_normalized[i, j])) + "%",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > cm_normalized.max() / 2. else "black")

    plt.tight_layout()
    plt.show()
    
    
def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def save_info(coco_evaluator, category_index, save_name, additional_info=""):
    iou_type = coco_evaluator.params.iouType
    print(f"mAP metric: {iou_type}")
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_evaluator)

    # calculate voc info for every classes(IoU=0.5)
    classes = [v for v in category_index.values() if v != "N/A"]
    voc_map_info_list = []
    for i in range(len(classes)):
        stats, _ = summarize(coco_evaluator, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(classes[i], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    # 将验证结果保存至txt文件中
    with open(save_name, "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        
        if additional_info:
            record_lines.extend(["", additional_info])
            
        f.write("\n".join(record_lines))


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # 计算交集的坐标
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    # 计算交集的面积
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # 计算两个框的面积
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    # 计算并集的面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area

    return iou


def normalize_confusion_matrix(cm):
    """归一化混淆矩阵的每一行，使其总和为1"""
    row_sums = cm.sum(axis=1, keepdims=True)
    return cm / row_sums


def move_background_to_end(cm, class_names):
    # 将第一行移到最后
    cm = np.vstack((cm[1:], cm[0]))
    
    # 将第一列移到最后
    cm = np.hstack((cm[:, 1:], cm[:, 0:1]))

    # 调整类别标签
    class_names = class_names[1:] + [class_names[0]]

    return cm, class_names


def main(parser_data):
    
    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"
    
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    
    delete_all_files("/root/autodl-tmp/new_swin/Detection_Baseline_swin/confusion_matrix")

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # read class_indict
    label_json_path = parser_data.label_json_path
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        category_index = json.load(f)

    data_root = parser_data.data_path

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    val_dataset = CocoDetection(data_root, "test", data_transform["val"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=val_dataset.collate_fn)

    # create model
    backbone = swin_fpn_backbone("swin_tiny","/root/autodl-tmp/new_swin/Detection_Baseline_swin/Weight/swin_tiny.pth")
    # backbone = resnet50_fpn_backbone(pretrain_path="Weight/resnet50.pth", trainable_layers=3)
    model = MaskRCNN(backbone, num_classes=args.num_classes + 1)

    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # print(model)

    model.to(device)

    # evaluate on the val dataset
    cpu_device = torch.device("cpu")

    det_metric = EvalCOCOMetric(val_dataset.coco, "bbox", "det_results.json")
    seg_metric = EvalCOCOMetric(val_dataset.coco, "segm", "seg_results.json")
    
    miou_list = []
    # For collecting true and predicted labels
    true_labels = []
    predicted_labels = []
    
    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            det_metric.update(targets, outputs)
            seg_metric.update(targets, outputs)
            
             # Collect true and predicted labels
            for target, output in zip(targets, outputs):
                true_labels_list = target['labels'].tolist()
                pred_labels_list = output.get('labels', []).tolist()

                # 如果真实标签的数量大于预测标签的数量
                while len(pred_labels_list) < len(true_labels_list):
                    pred_labels_list.append(0)  # 增加背景类标签

                # 如果预测标签的数量大于真实标签的数量
                while len(true_labels_list) < len(pred_labels_list):
                    true_labels_list.append(0)  # 增加背景类标签

                # 现在，真实标签和预测标签的数量应该是相同的
                for true_label, pred_label in zip(true_labels_list, pred_labels_list):
                    true_labels.append(true_label)
                    predicted_labels.append(pred_label)
                

    det_metric.synchronize_results()
    seg_metric.synchronize_results()
    det_metric.evaluate()
    seg_metric.evaluate()
    

    average_miou = np.mean(miou_list)

    save_info(det_metric.coco_evaluator, category_index, "det_record_0.4_mAP.txt", additional_info=f"mIoU: {average_miou:.3f}")
    save_info(seg_metric.coco_evaluator, category_index, "seg_record_0.4_mAP.txt", additional_info=f"mIoU: {average_miou:.3f}")
    
    # evaluate on the test dataset
    det_info, seg_info = utils.evaluate(model, val_dataset_loader, device=device)

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
    
    # Compute and plot the confusion matrix

    cm = compute_confusion_matrix(true_labels, predicted_labels, len(category_index))
    # cm_adjusted, class_names_adjusted = move_background_to_end(cm, class_names)
    
    saved_path = "/root/autodl-tmp/new_swin/Detection_Baseline_swin/confusion_matrix"
    
    plot_confusion_matrix(cm, list(category_index.values()), cmap=plt.cm.Blues)
    plt.savefig("/root/autodl-tmp/new_swin/Detection_Baseline_swin/confusion_matrix/confusion_matrix.png")

    print(f"Confusion matrix saved to: {saved_path}")
    # print(f"Average mIoU: {average_miou:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', type=int, default=16, help='number of classes')

    # 数据集的根目录
    parser.add_argument('--data-path', default='/root/autodl-tmp/new_swin/Detection_Baseline_swin/data_new/coco', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights-path', default='/root/autodl-tmp/new_swin/Detection_Baseline_swin/save_weights/cbam_1/model_29.pth', type=str, help='training weights')

    # batch size(set to 1, don't change)
    parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                        help='batch size when validation.')
    # 类别索引和类别名称对应关系
    parser.add_argument('--label-json-path', type=str, default="coco91_indices.json")

    args = parser.parse_args()

    main(args)
