import os
import numpy as np
from tqdm import tqdm
import glob
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_images_dir',  default='neocr_dataset/Images/users/pixtract/dataset/')
parser.add_argument('--input_annotations_dir', default='neocr_dataset/Annotations/users/pixtract/dataset/')
parser.add_argument('--save_img_dir', default='extracted_images/')
parser.add_argument('--save_xml_dir', default='extracted_annotations/')
args = parser.parse_args()


useless_list = [93519188, 105215353, 106907413, 140286525, 209596332, 230664808, 247632908, 328242995, 
                354001143, 377854318, 389227879, 407877135, 416780208, 423115341, 433780658, 436020480, 
                460235797, 491350693, 494481392, 528495588, 529587527, 539233740, 575687816, 586185068, 
                596597607, 616541433, 630994834, 642970880, 657467816, 692711963, 706795565, 708094209, 
                721270757, 833382544, 926016157, 291305860, 430641252, 496668027, 507940877, 662544171]

useless_images = []
for number in useless_list:
    image_name = f"img_{number}.jpg"
    useless_images.append(image_name)

images_path = args.input_images_dir
image_files = glob.glob(f"{images_path}/*.jpg")
image_files.sort()

annotations_path = args.input_annotations_dir
xml_files = glob.glob(f"{annotations_path}/*.xml")
xml_files.sort()

save_img_dir = args.save_img_dir
os.makedirs(save_img_dir, exist_ok=True)
save_xml_dir = args.save_xml_dir
os.makedirs(save_xml_dir, exist_ok=True)

EXTRACT = 0
print('Making Extracted Images...')
for (image_file, xml_file) in zip(image_files, xml_files):
    
    img_filename = os.path.basename(image_file)
    xml_filename = os.path.basename(xml_file)
    
    if img_filename in useless_images:
        EXTRACT += 1
        continue
        
    shutil.copy(image_file, f'{save_img_dir}/{img_filename}')
    shutil.copy(xml_file, f'{save_xml_dir}/{xml_filename}')

print(f'Useless Images {EXTRACT}')

