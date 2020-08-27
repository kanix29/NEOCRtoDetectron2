import os
import numpy as np
import cv2
from utils import get_neocr_dicts
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_images_dir',  default='neocr_dataset/Images/users/pixtract/dataset/')
parser.add_argument('--input_annotations_dir', default='neocr_dataset/Annotations/users/pixtract/dataset/')
parser.add_argument('--save_dir', default='annotated_images/')
args = parser.parse_args()

images_path = args.input_images_dir
annotations_path = args.input_annotations_dir

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

print('Loading Data...')
dataset_dicts = get_neocr_dicts(images_path, annotations_path)

print('Making Annotated Images...')
for index, d in tqdm(enumerate(dataset_dicts), total=len(dataset_dicts)):
    img = cv2.imread(d["file_name"])
    img_filename = os.path.basename(d["file_name"])

    for a in d['annotations']:
        xmin, ymin, xmax, ymax = a['bbox']
        img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,255,0),2)

    cv2.imwrite(f"{save_dir}/ann_{img_filename}", img)