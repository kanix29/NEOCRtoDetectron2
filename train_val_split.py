import glob, os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--percentage_val', default=20, help='percentage of val data')
args = parser.parse_args() 

img_input_path = 'neocr_dataset/Images/users/pixtract/dataset/'
ann_input_path = 'neocr_dataset/Annotations/users/pixtract/dataset/'

img_val_output_path = 'images/val/'
xml_val_output_path = 'annotations/val/'
img_train_output_path = 'images/train/'
xml_train_output_path = 'annotations/train/'

os.makedirs(img_val_output_path, exist_ok=True)
os.makedirs(xml_val_output_path, exist_ok=True)
os.makedirs(img_train_output_path, exist_ok=True)
os.makedirs(xml_train_output_path, exist_ok=True)

# Percentage of images to be used for the valid set
index_val = round(100 / args.percentage_val)

img_files = glob.glob(f"{img_input_path}/*.jpg")
xml_files = glob.glob(f"{ann_input_path}/*.xml")
img_files.sort()
xml_files.sort()

counter = 1
# Split train and valid data
print('During Splitting ...')
for img_pathAndFilename, xml_pathAndFilename in zip(img_files, xml_files):
    img_filename = os.path.basename(img_pathAndFilename)
    xml_filename = os.path.basename(xml_pathAndFilename)    

    if counter == index_val:
        shutil.copy(img_pathAndFilename, img_val_output_path+img_filename)
        shutil.copy(xml_pathAndFilename, xml_val_output_path+xml_filename)
        counter = 1
    else:
        shutil.copy(img_pathAndFilename, img_train_output_path+img_filename)
        shutil.copy(xml_pathAndFilename, xml_train_output_path+xml_filename)
        counter = counter + 1
    
print("Complete to split train and valid data")