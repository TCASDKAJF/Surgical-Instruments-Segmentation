import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.patches as patches

def get_filtered_dataframe(file_path, video, clip, step):
    df = pd.read_csv(file_path)
    df = df[(df['video'] == video) & (df['clip'] == clip) & (df['step'] == step) & (df['tool_key'] != 18)]  # 删除第18号类别
    df['tool_key'] = df['tool_key'].apply(lambda x: 20 if x in [2, 8] else x)  # 将类别2和类别8结合在一起
    return df

def generate_color_dict():
    color_dict = {
        0: [0, 0, 0],  # Background class
        0: [255, 0, 0],  # Tool: suction
        1: [0, 255, 0],  # New combined class for Tool: kerrisons_upcut (2) and Tool: kerrisons_downcut (8)
        2: [0, 0, 255],  # Tool: pituitary_rongeurs
        3: [255, 255, 0],  # Tool: retractable_knife
        4: [0, 255, 255],  # Tool: freer_elevator
        5: [255, 0, 255],  # Tool: spatula_dissector
        6: [128, 0, 0],  # Tool: dural_scissors
        7: [0, 0, 128],  # Tool: stealth_pointer
        8: [128, 128, 0],  # Tool: surgiflo
        9: [0, 128, 128],  # Tool: cup_forceps
        10: [128, 0, 128],  # Tool: ring_curette
        11: [192, 192, 192],  # Tool: cottle
        12: [128, 128, 128],  # Tool: drill
        13: [153, 153, 255],  # Tool: blakesley
        14: [153, 255, 153],  # Tool: bipolar_forceps
        15: [255, 153, 153],  # Tool: doppler
    }
    return color_dict

def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def generate_masks(df, output_dir, color_dict):
    previous_sec = -1
    mask = None
    tool_key_in_use = None

    image_path_template = 'G:/Final_Data/segmentation/images_test/video{video}_clip{clip}_step{step}/{sec}.jpg'

    for _, row in df.iterrows():
        sec = row['sec']
        tool_key = str(row['tool_key'])

        if tool_key == "none":
            continue

        if sec != previous_sec and previous_sec != -1 and mask is not None:
            mask_file = os.path.join(output_dir, f'{str(previous_sec).zfill(3)}_mask_tool_{tool_key_in_use}.png')
            cv2.imwrite(mask_file, mask)
            plt.imshow(mask)
            plt.title(f'Second {previous_sec} - Tool {tool_key_in_use}')
            plt.show()
            mask = None

        image_path = image_path_template.format(video=str(row["video"]).zfill(2),
                                                clip=str(row["clip"]).zfill(2),
                                                step=str(row["step"]).zfill(1),
                                                sec=str(sec).zfill(3))

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1], 3))
            tool_key_in_use = tool_key

        points = np.array(
            [[int(float(x)), int(float(y))] for x, y in (point.split(',') for point in row['points'].split(';'))])
        cv2.fillPoly(mask, [points], color_dict[int(tool_key)])
        previous_sec = sec

    # writing masks of the last sec
    if mask is not None:
        mask_file = os.path.join(output_dir, f'{str(sec).zfill(3)}_mask_tool_{tool_key_in_use}.png')
        cv2.imwrite(mask_file, mask)
        plt.imshow(mask)
        plt.title(f'Second {sec} - Tool {tool_key_in_use}')
        plt.show()

def mask_exists(output_dir, sec):
    for file in os.listdir(output_dir):
        if file.startswith(f'{str(sec).zfill(3)}_mask'):
            return True
    return False

def generate_all_masks(output_dir, image_shape, main_dir):
    # Create a mask for every image in the directory
    for image_file in os.listdir(main_dir):
        sec = int(image_file.split('.')[0])  # Get the frame number from the file name
        mask_file = os.path.join(output_dir, f'{str(sec).zfill(3)}_mask_tool_0.png')
        # If a mask with this timestamp already exists, don't overwrite it
        if not mask_exists(output_dir, sec):
            mask = np.zeros((image_shape[0], image_shape[1], 3))
            cv2.imwrite(mask_file, mask)

def check_and_create_empty_mask(main_dir, output_dir, image_shape):
    for image_name in os.listdir(main_dir):
        if not image_name.endswith(".jpg"):
            continue

        sec = image_name[:-4]  # Remove ".jpg" to get the sec
        mask_path = os.path.join(output_dir, f'{sec}_mask.png')

        # If the mask doesn't exist, create and save an empty one
        if not os.path.isfile(mask_path):
            mask = np.zeros((image_shape[0], image_shape[1], 3))
            cv2.imwrite(mask_path, mask)

def process_all_videos_in_directory(main_dir, file_path):
    color_dict = generate_color_dict()

    for subfolder in os.listdir(main_dir):
        if not os.path.isdir(os.path.join(main_dir, subfolder)):
            continue
        video, clip, step = subfolder.split('_')[0][5:], subfolder.split('_')[1][4:], subfolder.split('_')[2][4:]

        output_dir = f"G:/Final_Data/segmentation/images_masks/{subfolder}"  # Changed to your new output directory
        create_output_dir(output_dir)

        df = get_filtered_dataframe(file_path, int(video), int(clip), int(step))

        image_path = f'"G:\Final_Data\segmentation\images/{subfolder}/000.jpg"'  # Changed to your new images directory
        image_shape = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).shape

        # Generate masks for frames with labels
        generate_masks(df, output_dir, color_dict)

        # Generate empty masks for frames without labels
        generate_all_masks(output_dir, image_shape, main_dir=os.path.join(main_dir, subfolder))


def main():
    file_path = 'G:/Final_Data/segmentation/points_clean_none_removed.csv'  # The path to the cleaned CSV file
    main_dir = 'G:/Final_Data/segmentation/images'  # Update to your new images directory

    process_all_videos_in_directory(main_dir, file_path)

if __name__ == "__main__":
    main()