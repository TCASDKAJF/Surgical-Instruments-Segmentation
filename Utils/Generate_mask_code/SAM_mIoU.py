import os
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    if img is None:
        print(f"无法加载图片: {image_path}")
        return None
    return img / 255

def calculate_metrics(mask, prediction):
    mask_flat = mask.flatten()
    prediction_flat = prediction.flatten()
    iou_score = np.sum(np.logical_and(mask_flat, prediction_flat)) / np.sum(np.logical_or(mask_flat, prediction_flat))
    pixel_accuracy = np.mean(mask_flat == prediction_flat)
    precision = precision_score(mask_flat, prediction_flat)
    recall = recall_score(mask_flat, prediction_flat)
    return iou_score, pixel_accuracy, precision, recall

def calculate_metrics_over_dir(groundtruth_dir, prediction_dir):
    iou_total = 0
    pa_total = 0
    precision_total = 0
    recall_total = 0
    num_images = 0

    for filename in os.listdir(groundtruth_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            groundtruth = load_image(os.path.join(groundtruth_dir, filename))
            prediction = load_image(os.path.join(prediction_dir, filename.replace(".png", "_mask.png")))
            if groundtruth is None or prediction is None:
                continue
            iou_score, pixel_accuracy, precision, recall = calculate_metrics(groundtruth, prediction)
            print(f"For image {filename}, IoU: {iou_score}, Pixel Accuracy: {pixel_accuracy}, Precision: {precision}, Recall: {recall}")
            iou_total += iou_score
            pa_total += pixel_accuracy
            precision_total += precision
            recall_total += recall
            num_images += 1

    miou = iou_total / num_images
    mpa = pa_total / num_images
    avg_precision = precision_total / num_images
    avg_recall = recall_total / num_images
    return miou, mpa, avg_precision, avg_recall

if __name__ == "__main__":
    groundtruth_dir = r"G:\baseline\Binary_result\SAM\test_result1 (1)\test_result1\ground_truth"
    prediction_dir = r"G:\baseline\Binary_result\SAM\test_result1 (1)\test_result1\binary_mask"
    miou, mpa, avg_precision, avg_recall = calculate_metrics_over_dir(groundtruth_dir, prediction_dir)
    print(f"Mean Intersection over Union: {miou}")
    print(f"Mean Pixel Accuracy: {mpa}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")

