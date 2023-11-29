from PIL import Image
import os

def convert_bmp_to_jpg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of BMP files in the input folder
    bmp_files = [f for f in os.listdir(input_folder) if f.endswith('.bmp')]

    for bmp_file in bmp_files:
        # Create the input and output file paths
        input_path = os.path.join(input_folder, bmp_file)
        output_path = os.path.join(output_folder, os.path.splitext(bmp_file)[0] + '.jpg')

        # Open the BMP image
        with Image.open(input_path) as img:
            # Save the image as JPG
            img.convert('RGB').save(output_path, 'JPEG')

if __name__ == "__main__":
    input_folder = r"D:\Homework\Dataset - 複製\Q1_Image"
    output_folder = r"D:\Homework\Dataset - 複製\Q1_Image"

    convert_bmp_to_jpg(input_folder, output_folder)
