# NEOCR Dataset (Natural Environment OCR Dataset) to Detectron2

# Usage
```sh
wget http://www.iapr-tc11.org/dataset/NEOCR/neocr_dataset.tar.gz
tar -xvf neocr_dataset.tar.gz
```

split train and valida data
```python
python train_val_split.py 
```

# 学習はGoogle colabでやってモデルを作成。
そのモデルをインストールして、ローカルで予測をする。

## Environment


requirements.txtを書く必要がある

lxlm == 
detectron2 == 



[Link](http://www.iapr-tc11.org/mediawiki/index.php?title=NEOCR:_Natural_Environment_OCR_Dataset)

## Zip file
`Annotations/users/pixtract/dataset/*.xml`\
`Images/users/pixtract/dataset/*.jpg`\
The annotation is provided in XML based on the schema of LabelMe.

## xml2coco
[GitHub](https://github.com/aaronlelevier/mlearning/blob/master/mlearning/coco.py)

