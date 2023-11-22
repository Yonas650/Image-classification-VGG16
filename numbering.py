import os
import re

# Define the directory path
dir_path = "/Users/yonasmulu/Desktop/Arpro/val/shorts"

# List all files in the directory
files = os.listdir(dir_path)

# Filter image files with the naming format "imagename_numbering.jpg"
image_files = [file for file in files if re.search(".*_\d+\.(jpg|jpeg|png|gif|bmp|tiff)", file, re.IGNORECASE)]

# Sort the filtered image files based on the numbering part
image_files.sort(key=lambda x: int(re.findall("\d+", x)[0]))

# Initialize the new file numbering
counter = 1

# Iterate through the image files
for file in image_files:
    # Split the filename into name and extension
    file_name, file_ext = os.path.splitext(file)

    # Split the file name into imagename and numbering
    imagename, _ = file_name.rsplit("_", 1)

    # Construct the new file name with the new numbering
    new_file_name = f"{imagename}_{counter}"

    # Reconstruct the new file path with the new name and extension
    new_file_path = os.path.join(dir_path, f"{new_file_name}{file_ext}")

    # Rename the file
    os.rename(os.path.join(dir_path, file), new_file_path)

    # Increment the file numbering counter
    counter += 1

print(f"Renamed the numbering of {counter - 1} image files in the directory: {dir_path}")
