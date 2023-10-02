import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import cv2
import pandas as pd

import imgviz
import numpy as np

import labelme
from my_labelme import LabelFile as LF

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument("--input_dir", default="/data/result/changxiu/open_source/SAM/instrument_dataset/annotations_json/testing", help="input annotated directory")
    # parser.add_argument("--output_dir", default="/data/result/changxiu/open_source/SAM/instrument_dataset/coco_format/testing", help="output dataset directory")
    # parser.add_argument("--labels", default="F:\coco2017\val2017\labels.txt", help="labels file")
    parser.add_argument("--input_dir",
                        default=r"D:\BaiduNetdiskDownload\instrument_Dataset\annotations_json\validation",
                        help="input annotated directory")
    parser.add_argument("--output_dir",
                        default=r"D:\BaiduNetdiskDownload\instrument_Dataset\annotations_json\validation",
                        help="output dataset directory")

    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)

        os.makedirs(args.output_dir)
        os.makedirs(osp.join(args.output_dir, "JPEGImages"))

    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    # class_name_to_id = {}
    # for i, line in enumerate(open(args.labels).readlines()):
    #     class_id = i - 1  # starts with -1
    #     class_name = line.strip()
    #     if class_id == -1:
    #         assert class_name == "__ignore__"
    #         continue
    #     class_name_to_id[class_name] = class_id
    #     data["categories"].append(
    #         dict(supercategory=None, id=class_id, name=class_name, )
    #     )
    class_name_to_id = get_label_map()
    id_to_class_name = {}
    for class_name, class_id in class_name_to_id.items():
        data["categories"].append(dict(supercategory=None, id=class_id, name=class_name, ))
        id_to_class_name[class_id] = class_name


    out_ann_file = osp.join(args.output_dir, "annotations.json")
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    print(len(label_files), label_files[:10])
    for image_id, filename in enumerate(label_files):
        if filename.endswith('annotations_training.json'):
            continue
        print("Generating dataset from:", filename)

        # label_file = labelme.LabelFile(filename=filename)
        label_file = LF(filename=filename)
        # print(label_file)
        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")

        # img = labelme.utils.img_data_to_arr(label_file.imageData)
        # imgviz.io.imsave(out_img_file, img)
        img_name = label_file.imagePath
        print('img_name:', img_name)
        img = cv2.imread(img_name)
        # print(img.shape)
        img_height = label_file.imageHeight
        img_width = label_file.imageWidth
        img_shape = [img_height, img_width, 3]
        base = osp.splitext(osp.basename(img_name))[0]

        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(base + '.jpg'),
                height=img_height,
                width=img_width,
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        print('shapes:', len(label_file.shapes), label_file.shapes)
        if len(label_file.shapes):
            for shape in label_file.shapes:
                points = shape["points"]
                label_id = shape["label"]
                label = id_to_class_name[label_id]
                group_id = shape.get("group_id")
                shape_type = shape.get("shape_type", "polygon")
                mask = labelme.utils.shape_to_mask(
                    img_shape[:2], points, shape_type
                )

                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)
                print('instance:', instance)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask

                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                else:
                    points = np.asarray(points).flatten().tolist()

                segmentations[instance].append(points)
            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]

                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data["annotations"].append(
                    dict(
                        id=len(data["annotations"]),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )


            # if not args.noviz:
            #     labels, captions, masks = zip(
            #         *[
            #             # (cnm, id_to_class_name[cnm], msk)
            #             # for (cnm, gid), msk in masks.items()
            #             # if cnm in id_to_class_name
            #             (class_name_to_id[cnm], cnm, msk)
            #             for (cnm, gid), msk in masks.items()
            #             if cnm in class_name_to_id
            #
            #         ]
            #     )
            #     viz = imgviz.instances2rgb(
            #         image=img,
            #         labels=labels,
            #         masks=masks,
            #         captions=captions,
            #         font_size=15,
            #         line_width=2,
            #     )
            #     out_viz_file = osp.join(
            #         args.output_dir, "Visualization", base + ".jpg"
            #     )
            #     imgviz.io.imsave(out_viz_file, viz)


    with open(out_ann_file, "w") as f:
        json.dump(data, f)
    print('finish:', out_ann_file)
    print(data["annotations"][:10])


def csv_to_json():
    file_path = 'points_clean_none_removed.csv'
    # base_path = '/data/result/changxiu/open_source/SAM/instrument_dataset/annotations/training_label/'
    # save_base = base_path.replace('annotations', 'annotations_json')
    # img_base = base_path.replace('annotations', 'image').replace('_label', '')
    df = pd.read_csv(file_path)  # , nrows=100
    # print(df)
    label_info = df.values.tolist()

    ## 解析csv中的标注数据
    data_info = {}
    for dat in label_info:
        # dir_name = f"video{dat[0]:02d}_clip{dat[1]:02d}_step{dat[2]}"
        # file_name = f"video{dat[0]:02d}_clip{dat[1]:02d}_step{dat[2]}_{int(dat[3]):03d}_mask_tool_{int(dat[5])}.png"
        # img_name = f"video{dat[0]:02d}_clip{dat[1]:02d}_step{dat[2]}_{int(dat[3]):03d}.jpg"

        # print(file_name)
        points = dat[6].split(';')
        points = [m.split(',') for m in points]
        points = [[float(m[0]), float(m[1])] for m in points]
        # print(points)

        # img_path = os.path.join(img_base, dir_name, img_name)
        # if not os.path.exists(img_path):
        #     continue
        #
        # print('img', img_path)
        # img = cv2.imread(img_path)

        data = {}
        data['imagePath'] = None
        # data['flags'] = {}
        data['imageWidth'] = 1280
        data['imageHeight'] = 720
        data['imageData'] = None
        # data['version'] = "5.0.1"
        data["shapes"] = []

        itemData = {'points': []}
        itemData['points'].extend(points)
        # itemData["flag"] = {}
        # itemData["group_id"] = None
        itemData["shape_type"] = "polygon"
        itemData["label"] = int(dat[5])
        data["shapes"].append(itemData)

        json_name = f"video{dat[0]:02d}_clip{dat[1]:02d}_step{dat[2]}_{int(dat[3]):03d}.json"

        if json_name not in data_info.keys():
            data_info[json_name] = data
        else:
            data_info[json_name]["shapes"].append(itemData)

        # break

    # print(data_info)
    # base_path = '/data/result/changxiu/open_source/SAM/instrument_dataset/annotations/training_label/'
    # save_base = base_path.replace('annotations', 'annotations_json')
    # img_base = base_path.replace('annotations', 'image').replace('_label', '')

    img_base = '/data/result/changxiu/open_source/SAM/instrument_dataset/image/'
    datasets = ['training', 'validation', 'testing']
    save_base = img_base.replace('image', 'annotations_json')

    i = 0
    for dat_set in datasets:
        img_sets = os.listdir(img_base + dat_set + '/')
        for img_set in img_sets:
            img_files = os.listdir(os.path.join(img_base, dat_set, img_set))
            for img_f in img_files:
                img_path = os.path.join(img_base, dat_set, img_set, img_f)
                # img = cv2.imread(img_path)
                f_name = img_f.replace('.jpg', '.json')
                jsonPath = os.path.join(save_base, dat_set, img_set, f_name)
                os.makedirs(os.path.join(save_base, dat_set, img_set), exist_ok=True)
                if f_name in data_info.keys():
                    img_label = data_info[f_name]
                    img_label['imagePath'] = img_path
                else:
                    data = {}
                    data['imagePath'] = img_path
                    # data['flags'] = {}
                    data['imageWidth'] = 1280
                    data['imageHeight'] = 720
                    data['imageData'] = None
                    # data['version'] = "5.0.1"
                    data["shapes"] = []
                    img_label = data

                with open(jsonPath, "w") as f:
                    json.dump(img_label, f)
                print('save to:', jsonPath)

                i += 1
                # if i > 100:
                #     break

    print('finish! data num:', i)


def check_img_shape():
    img_base = r'D:\BaiduNetdiskDownload\instrument_Dataset\image\\'
    datasets = ['training', 'validation', 'testing']
    for dat in datasets:
        i = 0
        img_sets = os.listdir(img_base + dat + '/')
        for img_set in img_sets:
            img_files = os.listdir(os.path.join(img_base, dat, img_set))
            i += len(img_files)

            # for img_f in img_files:
            #     img_path = os.path.join(img_base, dat, img_set, img_f)
            #     img = cv2.imread(img_path)
            #     print(img.shape)

                # break
        print(dat, i)


def move_all_images():
    import shutil
    img_base = '/data/result/changxiu/open_source/SAM/instrument_dataset/annotations_json/'
    datasets = ['training', 'validation', 'testing']
    for dat in datasets:
        img_sets = os.listdir(img_base + dat + '/')
        for img_set in img_sets:
            img_files = os.listdir(os.path.join(img_base, dat, img_set))
            for img_f in img_files:
                img_path = os.path.join(img_base, dat, img_set, img_f)
                # img = cv2.imread(img_path)
                # print(img.shape)
                shutil.move(img_path, os.path.join(img_base, dat, img_f))

                # break

        img_sets = os.listdir(img_base + dat + '/')
        for img_set in img_sets:
            if os.path.isfile(os.path.join(img_base, dat, img_set)):
                continue
            else:
                os.rmdir(os.path.join(img_base, dat, img_set))


def get_label_map():
    """{'freer_elevator': 5, 'spatula_dissector': 6, 'kerrisons_upcut': 2, 'suction': 1, 'retractable_knife': 4,
     'dural_scissors': 7, 'pituitary_rongeurs': 3, 'surgiflo': 10, 'cup_forceps': 11, 'blakesley': 15,
     'ring_curette': 12, 'kerrisons_downcut': 8, 'cottle': 13, 'doppler': 17, 'drill': 14, 'bipolar_forceps': 16,
     'stealth_pointer': 9, 'irrigation_syringe': 18}
    """
    file_path = 'points_clean_none_removed.csv'
    df = pd.read_csv(file_path)  # , nrows=100
    # print(df)
    label_info = df.values.tolist()

    ## 解析csv中的标注数据
    label_map = {}
    for dat in label_info:
        class_name = dat[4]
        class_id = int(dat[5])
        if class_name in label_map.keys():
            continue
        else:
            label_map[class_name] = class_id

    # print(label_map)
    # label_map['background'] = 0

    return label_map


if __name__ == "__main__":
    # 1. csv_to_json
    # csv_to_json()
    # 2. move all images together
    # move_all_images()

    # 3. generate coco-format
    main()

    # filename = "D:/work file/open_source/lightning_sam/lightning_sam/preprocess_data.py"
    # print(os.path.relpath(filename))
    # print(os.path.splitext(osp.basename(filename)))






