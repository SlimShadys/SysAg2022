import torch.utils.data as data
import pandas as pd
import numpy as np
import librosa
import librosa.display
import random
import time
import os
import io
import matplotlib.pyplot as plt
from PIL import Image

from matplotlib import cm

def plotFigure(img):    
    plt.figure(figsize=(3,3))                                         # Set size of figure (300x300)
    plt.axis('off')                                                   # Remove axis
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])   # Remove the white edge
    
    plt.imshow(img)
    plt.show()                                                        # Show the figure and close it
    
    plt.close()

def getLabel(emozione):
    if (emozione == 'Disgusto'):
        label = 0
    elif (emozione == 'Gioia') :
        label = 1
    elif (emozione == 'Paura'):
        label = 2
    elif (emozione == 'Rabbia'):
        label = 3
    elif (emozione == 'Sorpresa'):
        label = 4
    elif (emozione == 'Tristezza'):
        label = 5
    else : # Neutrale
        label = 6
    return label

def getEmotion(label):
    if (label == 0):
        emozione = 'Disgusto'
    elif (label == 1) :
        emozione = 'Gioia'
    elif (label == 2):
        emozione = 'Paura'
    elif (label == 3):
        emozione = 'Rabbia'
    elif (label == 4):
        emozione = 'Sorpresa'
    elif (label == 5):
        emozione = 'Tristezza'
    else : # Neutrale
        emozione = 'Neutrale'
    return emozione 

class Demos(data.Dataset):

    def __init__(self, gender, split='train', transform=None, withAugmentation=True):
        
        if(os.getcwd().endswith("dataloader")):
            datasetDirectory = "../../Datasets"
        else:
            datasetDirectory = "../Datasets"
        dataset = "wav_DEMoS"

        csvName = "demos"
        
        self.transform = transform
        self.split = split
        self.gender = gender
        self.audios = os.path.join(datasetDirectory, dataset).replace('\\','/')

        if self.split == "train":
            if(gender == 'male'):       # CSV Uomo
                csv = pd.read_csv("{}/train_{}_male.csv".format(datasetDirectory, csvName), sep=";", encoding='UTF8', index_col=False)
            elif(gender == 'female'):   # CSV Donna
                csv = pd.read_csv("{}/train_{}_female.csv".format(datasetDirectory, csvName), sep=";", encoding='UTF8', index_col=False)
            else:                       # CSV Generale
                csv = pd.read_csv("{}/train_{}_all.csv".format(datasetDirectory, csvName), sep=";", encoding='UTF8', index_col=False)

            # Controlliamo se dobbiamo utilizzare i file di data augmentation o meno    
            if not withAugmentation:
                csv = csv[~csv['NOME_FILE'].str.contains('/{}_augmentation'.format(dataset), na=False)]

            csv.reset_index(drop=True, inplace=True)

            self.data = csv
                
        elif self.split == "val":
            if(gender == 'male'):       # CSV Uomo
                self.data = pd.read_csv("{}/val_{}_male.csv".format(datasetDirectory, csvName), sep=";", encoding='UTF8', index_col=False)
            elif(gender == 'female'):   # CSV Donna
                self.data = pd.read_csv("{}/val_{}_female.csv".format(datasetDirectory, csvName), sep=";", encoding='UTF8', index_col=False)
            else:                       # CSV Generale
                self.data = pd.read_csv("{}/val_{}_all.csv".format(datasetDirectory, csvName), sep=";", encoding='UTF8', index_col=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if(os.getcwd().endswith("dataloader")):
            datasetDirectory = "../../Datasets"
        else:
            datasetDirectory = "../Datasets"
        dataset = "wav_DEMoS"
        
        # Lettura della label associata al file audio nel CSV
        label = self.data.loc[idx, "EMOZIONE"]
        label = getLabel(label)
       
        # Lettura del percorso del file audio dal CSV
        audio_file = datasetDirectory + "/{}".format(dataset) + self.data.loc[idx, "NOME_FILE"]
        
        # Lettura del file .wav
        y, sr = librosa.load(audio_file, sr = 16000) # Use the default sampling rate of 22,050 Hz
        
        # Pre-emphasis filter
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

        # Creazione spettrogramma e salvataggio in IMG
        M = librosa.feature.melspectrogram(y=y, sr=sr, 
                                           fmax = sr/2,     # Maximum frequency to be used on the on the MEL scale
                                           n_fft=2048, 
                                           hop_length=512, 
                                           n_mels = 96,     # As per the Google Large-scale audio CNN paper
                                           power = 2)       # Power = 2 refers to squared amplitude
        

        fig = plt.Figure()
        plt.figure(figsize=(3,3))                                         # Set size of figure (300x300)
        plt.axis('off')                                                   # Remove axis
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])   # Remove the white edge
                
        # Power in DB
        img = librosa.power_to_db(M, ref=np.max)            # Covert to dB (log) 

        librosa.display.specshow(img, cmap=cm.jet)                        # Save the spectrogram
        
        fig = plt.gcf()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        #im = Image.open(buf)
        
        pil = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        
        plt.cla()
        plt.close()
        
        if self.transform is not None:
            pil = self.transform(pil)

        return {'image': pil, 'label': label}

if __name__ == "__main__":
    split = "train"
    gender = "male"
    demos_train = Demos(gender=gender,split=split,withAugmentation=True)
    print("Demos {} set successfully loaded".format(split))
    print("Loaded a total of {} samples".format(len(demos_train)))

    for i in range(5):
        
        random.seed(time.process_time())
        i = np.random.randint(len(demos_train))
        print("Label associata al file n. {}: {} ({})".format(i, demos_train[i]['label'], getEmotion(demos_train[i]['label'])))
        plotFigure(demos_train[i]['image'])