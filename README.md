# Integrative Segmentation Framework for Enhanced Tool Recognition in Pituitary Surgery

This repository accompanies my Master's Thesis. It provides the implementation details and necessary resources for the segmentation framework discussed.

## **Contents**

- [Binary Segmentation](#binary-segmentation)
- [Instance Segmentation](#instance-segmentation)
- [Our Proposed Method](#our-proposed-method)
- [Resources](#resources)

## **Binary Segmentation**

As detailed in the thesis, this project encompasses several methods for binary segmentation:

- **Unet**
- **SAM**
- **Mixmatch**

## **Instance Segmentation**

For instance segmentation, this repository contains implementations of several state-of-the-art (SOTA) models that serve as baselines:

- **Yolo**
- **mmdetection**
  - maskrcnn-resnet
  - htc

## **Our Proposed Method**

Our innovative contributions include:

- **Improved StyleGAN3 Code**
- **CME Code**
- **Enhanced MaskRCNN Code**

For each model and method, the required environment can be set up using the `requirements.txt` file located in the respective folders.

## **Resources**

- **Pre-trained Weights and Data**: All backbone pre-trained weights, trained weights for different models, and test data can be found [here](https://drive.google.com/drive/folders/1jhEfuKI__m2tYJaJNwF8zpjS2_wxn4Yw?usp=drive_link).
  
  - **Best Model**: Our top-performing model can be found at: `.\pth\all_in\best_all_in.pth"`
  
  - **Recording**: The aforementioned link also includes a recording showcasing the performance of the best model's weights.
