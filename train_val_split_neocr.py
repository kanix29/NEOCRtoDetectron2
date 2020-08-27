import glob, os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_images_dir',  default='neocr_dataset/Images/users/pixtract/dataset/')
parser.add_argument('--input_annotations_dir', default='neocr_dataset/Annotations/users/pixtract/dataset/')
parser.add_argument('--save_train_img_dir', default='images/train/')
parser.add_argument('--save_train_xml_dir', default='annotations/train/')
parser.add_argument('--save_val_img_dir', default='images/val/')
parser.add_argument('--save_val_xml_dir', default='annotations/val/')
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

img_input_path = args.input_images_dir
ann_input_path = args.input_annotations_dir

img_files = glob.glob(f"{img_input_path}/*.jpg")
xml_files = glob.glob(f"{ann_input_path}/*.xml")
img_files.sort()
xml_files.sort()

save_train_img_dir = args.save_train_img_dir
save_val_img_dir = args.save_val_img_dir
save_train_xml_dir = args.save_train_xml_dir
save_val_xml_dir = args.save_val_xml_dir

os.makedirs(save_train_img_dir, exist_ok=True)
os.makedirs(save_val_img_dir, exist_ok=True)
os.makedirs(save_train_xml_dir, exist_ok=True)
os.makedirs(save_val_xml_dir, exist_ok=True)

# splilt train and val data
num_extract = 0
print('Making Extracted Images...')
for (img_file, xml_file) in zip(img_files, xml_files):
    
    img_filename = os.path.basename(img_file)
    xml_filename = os.path.basename(xml_file)
    
    # add dataset annotated incorrectly to val directory
    if img_filename in useless_images:
        shutil.copy(img_file, f'{save_val_img_dir}/{img_filename}')
        shutil.copy(xml_file, f'{save_val_xml_dir}/{xml_filename}')
        num_extract += 1

    else:
        shutil.copy(img_file, f'{save_train_img_dir}/{img_filename}')
        shutil.copy(xml_file, f'{save_train_xml_dir}/{xml_filename}')

print(f'Useless Images {num_extract}')
print("Complete to split train and valid data")