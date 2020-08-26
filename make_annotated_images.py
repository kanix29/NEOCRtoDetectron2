import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import get_neocr_dicts
from tqdm import tqdm
import argparse

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

parser = argparse.ArgumentParser()
parser.add_argument('--images_dir',  default='neocr_dataset/Images/users/pixtract/dataset/')
parser.add_argument('--annotations_dir', default='neocr_dataset/Annotations/users/pixtract/dataset/')
parser.add_argument('--save_dir', default='annotated_images/')
args = parser.parse_args()

images_path = args.images_dir
annotations_path = args.annotations_dir

print('Loading Data...')
dataset_dicts = get_neocr_dicts(images_path, annotations_path)

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

## main
print('Making Annotated Images...')
for index, d in tqdm(enumerate(dataset_dicts), total=len(dataset_dicts)):
    img = cv2.imread(d["file_name"])
    img_filename = os.path.basename(d["file_name"])

    for a in d['annotations']:
        xmin, ymin, xmax, ymax = a['bbox']
        img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,255,0),2)

    cv2.imwrite(f"{save_dir}/ann_{img_filename}", img)