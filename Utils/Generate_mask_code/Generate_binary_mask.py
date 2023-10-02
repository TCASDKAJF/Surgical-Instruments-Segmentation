import pandas as pd
import numpy as np
import cv2
import os


def get_filtered_dataframe(file_path, video, clip, step):
    df = pd.read_csv(file_path)
    df = df[(df['video'] == video) & (df['clip'] == clip) & (df['step'] == step)]
    return df


def generate_color_dict():
    color_dict = {
        0: [0, 0, 0],
        1: [255, 255, 255]
    }
    return color_dict


def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def generate_masks(df, output_dir, color_dict, image_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])  # get sorted image files
    image_path_template = 'G:/Final_Data/segmentation/images/video{video}_clip{clip}_step{step}/video{video}_clip{clip}_step{step}_{sec}.jpg'

    for image_file in image_files:
        sec = int(image_file.split('_')[-1].split('.')[0])
        mask_file = os.path.join(output_dir, image_file.replace('.jpg',
                                                                '.png'))  # Use original image file name to create mask file name
        row = df[df['sec'] == sec]
        if len(row) > 0:
            row = row.iloc[0]
            image_path = image_path_template.format(video=str(row["video"]).zfill(2),
                                                    clip=str(row["clip"]).zfill(2),
                                                    step=str(row["step"]).zfill(1),
                                                    sec=str(sec).zfill(3))
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                mask = np.zeros((image.shape[0], image.shape[1], 3))  # initialize mask here
                points = np.array([[int(float(x)), int(float(y))] for x, y in
                                   (point.split(',') for point in row['points'].split(';'))])
                cv2.fillPoly(mask, [points], color_dict[1])
            else:
                mask = np.zeros((1080, 1920, 3))  # initialize mask here with default size
        else:
            mask = np.zeros((1080, 1920, 3))  # initialize mask here with default size

        mask = (mask > 0).astype(np.uint8) * 255  # Binary mask
        cv2.imwrite(mask_file, mask)


def process_all_videos_in_directory(main_dir, file_path):
    color_dict = generate_color_dict()

    for subfolder in os.listdir(main_dir):
        if not os.path.isdir(os.path.join(main_dir, subfolder)):
            continue
        video, clip, step = subfolder.split('_')[0][5:], subfolder.split('_')[1][4:], subfolder.split('_')[2][4:]
        image_dir = os.path.join(main_dir, subfolder)

        output_dir = f"G:/Final_Data/segmentation/images_binary_masks/{subfolder}"
        create_output_dir(output_dir)

        df = get_filtered_dataframe(file_path, int(video), int(clip), int(step))

        generate_masks(df, output_dir, color_dict, image_dir)


def main():
    file_path = 'G:/Final_Data/segmentation/points_clean_none_removed.csv'
    main_dir = 'G:/Final_Data/segmentation/images'

    process_all_videos_in_directory(main_dir, file_path)


if __name__ == "__main__":
    main()
