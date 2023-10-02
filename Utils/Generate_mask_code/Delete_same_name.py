import os

def delete_duplicate_files(directory):
    # List all the files in the directory
    for root, dirs, files in os.walk(directory):
        # Group files by sequence number
        files_dict = {}
        for file in files:
            if file.endswith('.png'):  # ensure we're only dealing with .png files
                sequence_number = file.split('_')[0]  # Get the sequence number
                if sequence_number not in files_dict:
                    files_dict[sequence_number] = []
                files_dict[sequence_number].append(file)

        # Go through each group
        for sequence_number, file_group in files_dict.items():
            if len(file_group) > 1:  # There is more than one file with this sequence number
                # Sort the files so that the non-tool ones come first
                file_group.sort(key=lambda x: "_tool" in x, reverse=True)
                # Delete the non-tool file
                os.remove(os.path.join(root, file_group[1]))
                print(f"Deleted {os.path.join(root, file_group[1])}")

# Use the function on your directory
delete_duplicate_files("G:/Final_Data/segmentation/images_2_masks")