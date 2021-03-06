# NEOCR (Natural Environment OCR) dataset to Detectron2's dataset

## Explanation
The Annotation of [NEOCR Dataset]((http://www.iapr-tc11.org/mediawiki/index.php?title=NEOCR:_Natural_Environment_OCR_Dataset))
is provided in XML based on the schema of LabelMe.\
`get_detectron_dict.get_neocr_dicts` is the function of converting LabelMe format to Detectron2 format. Therefore, copy and use this function.

## Folder structure
The following shows basic folder structure.
```
├── get_detectron_dict.py
│
├── Detectron2_NEOCR.ipynb
├── train_val_split.py 
├── text_detection_NEOCR.py 
├── extract_dataset.py 
├── annotate_images.py 
│
├── neocr_dataset (download)
│   ├── Annotations/...
│   └── Images/...
│
├── images (train_val_split.py)
│   ├── train/*.jpg
│   └── val/.jpg
└── annotaions (train_val_split.py)
     ├── train/*.xml
     └── val/*.cml
```

## Development Environment
* xlmtodict == 0.12.0
* lxlm == 4.5.0
* tqdm == 4.48.2
* torch == 1.6.0
* torchvision == 0.7.0
* detectron2 == 0.2.1
* opencv-python == 4.4.0.42
* numpy == 1.19.1
* matplotlib == 3.3.1

## Usage
Download dataset:
```sh
wget http://www.iapr-tc11.org/dataset/NEOCR/neocr_dataset.tar.gz
tar -xvf neocr_dataset.tar.gz
```
Split train and valid data:\
Add dataset annotated incorrectly to `val directory`
```sh
python train_val_split_neocr.py 
```
Get annotated images of valid dataset:
```sh
python text_detection_NEOCR.py -r [input_dir]
```

### Extra
Confirm images annotated correctly:
```sh
python annotate_images.py
```
Get dataset annotated correctly:
```sh
python extract_images.py
```

## About Notebook
`Detectron2_NEOCR.ipynb` is a explanation of the usage of detectron2.\
You can refer this notebook to train NEOCR dataset. Then, install `model_final.pth` and detect on local environment.
