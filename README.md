# NEOCR Dataset (Natural Environment OCR Dataset) to Detectron2

## Introduction


## Folder structure
The following shows basic folder structure.
```
├── utils.py
├── neocr_dataset
│   ├── Annotations/...
│   └── Images/...
│
├── train_val_split.py 
├── images 
│   ├── train/*.jpg
│   └── val/.jpg
├── annotaions
     ├── train/*.xml
     └── val/*.cml
```

## Development Environment
* lxlm == 4.5.0
* tqdm == 4.48.2
* torch == 1.6.0
* torchvision == 0.7.0
* detectron2 == 0.2.1
* opencv-python == 4.4.0.42
* numpy == 1.19.1
* matplotlib == 3.3.1

## Usage
### Download Datasets
```sh
wget http://www.iapr-tc11.org/dataset/NEOCR/neocr_dataset.tar.gz
tar -xvf neocr_dataset.tar.gz
```

### Split train and valid data
```python
python train_val_split.py 
```

### 学習はGoogle colabでやってモデルを作成。
そのモデルをインストールして、ローカルで予測をする。

[Link](http://www.iapr-tc11.org/mediawiki/index.php?title=NEOCR:_Natural_Environment_OCR_Dataset)

## Zip file
`Annotations/users/pixtract/dataset/*.xml`\
`Images/users/pixtract/dataset/*.jpg`\
The annotation is provided in XML based on the schema of LabelMe.

## xml2coco
[GitHub](https://github.com/aaronlelevier/mlearning/blob/master/mlearning/coco.py)

