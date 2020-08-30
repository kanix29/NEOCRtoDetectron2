import numpy as np
import os, json, cv2, random, glob
import argparse
from tqdm import tqdm
import torch, torchvision
import warnings
warnings.simplefilter('ignore', UserWarning)

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, default='output/text_detection_images')
    parser.add_argument('-m', '--model_path', type=str, default='model/neocr_model_final_v2.pth')
    return parser.parse_args()


class TextDetection():
    def __init__(self, input_dir, output_dir, model_path):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) # detect only box / Not use mask images
        self.cfg.DATASETS.TRAIN = ("neocr_train",)
        self.cfg.DATASETS.TEST = ("neocr_val", )
        self.cfg.DATALOADER.NUM_WORKERS = 2 # default 4
        self.cfg.SOLVER.IMS_PER_BATCH = 2 # default 16
        self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR / default 0.001
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (text)

        self.cfg.MODEL.DEVICE = 'cpu' # Use CPU on Local
        # self.cfg.OUTPUT_DIR = output_dir
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set a custom testing threshold

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_path = model_path

        MetadataCatalog.get("neocr_train").set(thing_classes=["text"]) # 画像にtextと出力させる用

    def prediction(self, img):
        predictor = DefaultPredictor(self.cfg)
        outputs = predictor(img)
        return outputs

    def annotation(self):
        # directory 内のファイルをすべて探索
        for pathAndFilename in tqdm(glob.iglob(f'{self.input_dir}/*'), total=len(os.listdir(self.input_dir))):
            # Skip not image files
            img_filename, ext = os.path.splitext(os.path.basename(pathAndFilename))
            if not ext in ['.jpg', '.JPG', '.png']:
                continue

            img = cv2.imread(pathAndFilename)
            
            outputs = self.prediction(img)
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"])
            cv2.imwrite(f'{self.output_dir}/annotated_{img_filename}{ext}', out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    # Define initial values
    args = parse_args()
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    MODEL_PATH = args.model_path

    if not os.path.isdir(INPUT_DIR):
        raise ValueError(f'the argment [-r] must be a directory')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    detector = TextDetection(INPUT_DIR, OUTPUT_DIR, MODEL_PATH)
    detector.annotation()
