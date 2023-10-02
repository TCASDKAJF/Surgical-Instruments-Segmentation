import os

def rename_files_in_directory(directory_path, subfolder):
    for filename in os.listdir(directory_path):
        if filename.endswith("_mask.png"):
            new_filename = f"{subfolder}_{filename.replace('_mask', '')}"
            os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_filename))

def main():
    parent_directory_path = "G:/Final_Data/segmentation/images_binary_masks"
    for subfolder in os.listdir(parent_directory_path):
        subfolder_path = os.path.join(parent_directory_path, subfolder)
        if os.path.isdir(subfolder_path):
            rename_files_in_directory(subfolder_path, subfolder)

if __name__ == "__main__":
    main()
