import os
import re

# Define the directory path
dir_path = "/Users/yonasmulu/Desktop/Arpro/val/Hoodies"

# List all files in the directory
files = os.listdir(dir_path)

# Filter image files that have "sweatshirts" in their name
image_files = [file for file in files if re.search("sweatshirts.*\.(jpg|jpeg|png|gif|bmp|tiff)", file, re.IGNORECASE)]

# Sort the filtered image files
image_files.sort()

# Initialize the new file numbering
counter = 1

# Iterate through the image files
for file in image_files:
    # Split the filename into name and extension
    file_name, file_ext = os.path.splitext(file)

    # Replace "sweatshirts" with "hoodies" in the name
    new_file_name = file_name.replace("sweatshirts", "hoodies")

    # Add the new file numbering to the name
    new_file_name = f"{new_file_name[:-len(str(counter-1))]}{counter}"

    # Reconstruct the new file path with the new name and extension
    new_file_path = os.path.join(dir_path, f"{new_file_name}{file_ext}")

    # Rename the file
    os.rename(os.path.join(dir_path, file), new_file_path)

    # Increment the file numbering counter
    counter += 1

print(f"Renamed {counter - 1} image files in the directory: {dir_path}")
