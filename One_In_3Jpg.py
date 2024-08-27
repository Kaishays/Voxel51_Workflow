import os
import shutil

def FindAllJpgs(directoryToSearchForJpg):
    filepaths = []
    count = 0
    for root, dirs, files in os.walk(directoryToSearchForJpg):
        for file in files:
            if file.endswith(".jpg"):
                count += 1
                print("found jpg: " + str(count))
                filepaths.append(os.path.join(root, file))
    return filepaths

def select_one_in_three_images(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Get a list of all files in the source directory
    filePaths = FindAllJpgs(source_dir)

    accumulated_index = 0.0
    n = 2.38
    count = 0 
    for file_path in filePaths:
        accumulated_index += 1
        if accumulated_index >= n:  # If accumulated index reaches or exceeds the threshold
            # Since file_path already includes the directory, don't join with source_dir again
            destination_path = os.path.join(destination_dir, os.path.basename(file_path))
            shutil.copy2(file_path, destination_path)
            count += 1
            print(count)
            accumulated_index -= n  # Reset the accumulated index by subtracting n

# Example usage
source_directory = "C:/Git/ml/DataManagement/datasets/06_V5/runs/slice_coco/labels_images_640_001" 
destination_directory = "C:/Git/ml/DataManagement/datasets/06_V5_Temp/oneIN3"  

select_one_in_three_images(source_directory, destination_directory)
