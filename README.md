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
###### The files made by the spectrograms augmentation script are already included in the downloaded Datasets. If you want to make new spectrograms, just delete the "{datasetName}_augmentation" folder in each Dataset folder and run this script.
```
python spectrograms_aug.py    # --> Make spectrograms
python makeCSV.py             # --> Update CSVs
```

# Training

## Start training with custom params
```
cd ser-experiment/
python train.py --dataset demos --gender male --attention bam --uses_drive True --withAugmentation False --batch_size 64 --epoch 71 --patience 10 etc.
```

# Testing

## Start testing with custom params
```
cd ser-experiment/
python test.py --dataset demos --gender male --attention bam --uses_drive True --batch_size 64 --loadModel ../Results/Models/Gruppo_A/best_model_male-epoch_68.pt
```

- Gianmarco Scarano
- Vincenzo Pio Sgarro
- Afrim Sokoli
__________________________________________________________________
