# FKEGW

Yuxiang Hong, Kai Lin, Jing Xu, Shengyong Li, Baohua Chang, and Dong Du, Expert knowledge-guided deep neural network based on context-aware hierarchy for foils joining quality monitoring, Advanced Engineering Informatics, 2025.


# Pytorch
* Pytorch implementation of FKEGW

## Requirements

* Python 3.11
* Pytorch 2.5.1
* torchvision

## Data Preprocessing
* Put the data to `Data/` folder.

## Train
```
python train.py
```
* You can find the detailed metrics during training process in the `'logs/'` folder and the model file in the `'Checkpoints/'` folder.

## Test
```
python test.py
```

* You can find the results in the `'Outputs/'` folder.
