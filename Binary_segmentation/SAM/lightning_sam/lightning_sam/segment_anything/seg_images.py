import shutil
from predictor import SamPredictor
from build_sam import sam_model_registry
import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from skimage import measure


def get_max_two_region(mask):
    """保留二值图像的最大连通域.
    """
    # 连通域标注
    mask_label = measure.label(mask, connectivity=1)
    # 获取连通域属性，包括面积、周长、质心等
    regions = measure.regionprops(mask_label)
    # 取所有连通域的面积
    area = [[region.label, region.area, region.bbox] for region in regions]
    # print('origin area:', area)
    # 找到最大面积连通域的标号
    # larger_label = np.argmax(area) + 1

    area.sort(key=lambda y:y[1], reverse=True)
    # print(area)

    if len(area) == 1:
        mask1 = np.where(mask_label == area[0][0], 1, 0).astype(np.uint8)
        # mask2 = mask1
        return mask1, mask1, area[0][2], area[0][2]
    else:
        # 最大连通域置1，其他置0
        mask1 = np.where(mask_label == area[0][0], 1, 0).astype(np.uint8)
        mask2 = np.where(mask_label == area[1][0], 1, 0).astype(np.uint8)

        return mask1, mask2, area[0][2], area[1][2]


def change_img_names():
    base_path = './sketch2cloth/'
    f_list = os.listdir(base_path)
    save_path = './test_imgs/'
    os.makedirs(save_path, exist_ok=True)

    for idx, f in enumerate(f_list):
        shutil.copy(base_path + f, save_path + f"img_{str(idx)}.jpg")

def main():
    sam = sam_model_registry["vit_b"](checkpoint="./model/sam_vit_b_01ec64.pth")
    # sam = sam_model_registry["vit_h"](checkpoint="./model/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)

    # mask_generator = SamAutomaticMaskGenerator(sam)


    base_path = './test_imgs/'
    f_list = os.listdir(base_path)

    save_base = './test_result/'

    for f_name in f_list:
        if f_name != '2.jpg':
            continue
        print('process:', f_name)
        img = cv2.imread(base_path + f_name)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        masks, _, _ = predictor.predict()

        # masks = mask_generator.generate()
        # print(img.shape, masks.shape, np.unique(masks))

        # cv2.imwrite(save_base + f_name, 255*masks.transpose([1, 2, 0]))
        mask = 1 - masks[0, :, :]

        dst1, dst2, bbox1, bbox2 = get_max_two_region(mask)
        # print(dst1, dst2)

        dst1 = cv2.merge([dst1, dst1, dst1])
        # dst1 = dst1 * img + 255 * (1 - dst1)
        dst2 = cv2.merge([dst2, dst2, dst2])
        # dst2 = dst2 * img + 255 * (1 - dst2)

        cv2.imwrite(save_base + f_name.replace('.jpg', '_1.jpg'), dst1*255)
        cv2.imwrite(save_base + f_name.replace('.jpg', '_2.jpg'), dst2*255)

        # dst1 = dst1[bbox1[0]:bbox1[2], bbox1[1]:bbox1[3], :]
        # dst2 = dst2[bbox2[0]:bbox2[2], bbox2[1]:bbox2[3], :]
        #
        # # save_path = save_base + f_name.split('.jpg')[0] + '/'
        # # os.makedirs(save_path, exist_ok=True)
        #
        # cv2.imwrite(save_base + f_name.replace('.jpg', '_1.jpg'), dst1)
        # cv2.imwrite(save_base + f_name.replace('.jpg', '_2.jpg'), dst2)



        # break




if __name__ == "__main__":
    main()
    # change_img_names()