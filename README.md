# SysAg2022

# Requirements

## Required libraries
```
pip install opencv-python librosa gdown
```

# Data preparation

## Download datasets
```
python downloadDatasets.py
```
## Splitting
```
python makeCSV.py
```
## Data Augmentation
```
python spectrograms_aug.py
```

# Training

## Start training with custom params
```
cd ser-experiment/
python train.py --dataset demos --gender male --uses_drive True --batch_size 14 --epoch 71 --patience 10 etc.
```

- Gianmarco Scarano
- Vincenzo Pio Sgarro
- Afrim Sokoli
__________________________________________________________________