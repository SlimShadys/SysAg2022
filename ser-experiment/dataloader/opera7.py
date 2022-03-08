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

def getLabel(label, validation):
    if(validation == "gender"):
        if (label == 'Donna'):
            label = 0
        else : # Uomo
            label = 1
        return label
    else: # validation == 'emotion'
        if (label == 'Disgusto'):
            label = 0
        elif (label == 'Gioia') :
            label = 1
        elif (label == 'Paura'):
            label = 2
        elif (label == 'Rabbia'):
            label = 3
        elif (label == 'Sorpresa'):
            label = 4
        elif (label == 'Tristezza'):
            label = 5
        else : # Neutrale
            label = 6
        return label

def getEmotion(label, validation):
    if(validation == "gender"):
        if (label == 0):
            label = 'Donna'
        else : # Uomo
            label = 'Uomo'
        return label
    else: # validation == 'emotion'
        if (label == 0):
            label = 'Disgusto'
        elif (label == 1) :
            label = 'Gioia'
        elif (label == 2):
            label = 'Paura'
        elif (label == 3):
            label = 'Rabbia'
        elif (label == 4):
            label = 'Sorpresa'
        elif (label == 5):
            label = 'Tristezza'
        else : # Neutrale
            label = 'Neutrale'
        return label 

class Opera7(data.Dataset):

    def __init__(self, gender, validation=None, split='val', transform=None):
        if(os.getcwd().endswith("dataloader")):
            datasetDirectory = "../../Datasets"
        else:
            datasetDirectory = "../Datasets"

        dataset = 'OPERA7_wav'
        csvName = "opera7"

        self.transform = transform
        self.split = split
        self.gender = gender
        self.validation = validation
        self.audios = os.path.join(datasetDirectory, dataset).replace('\\','/')

        if self.split == "val":
            if(gender == 'male'):       # CSV Uomo
                csv = pd.read_csv("{}/val_{}_male.csv".format(datasetDirectory, csvName), sep=';', encoding='UTF8', index_col=False)
            elif(gender == 'female'):   # CSV Donna
                csv = pd.read_csv("{}/val_{}_female.csv".format(datasetDirectory, csvName), sep=';', encoding='UTF8', index_col=False)
            else:                       # CSV Generale
                csv = pd.read_csv("{}/val_{}_all.csv".format(datasetDirectory, csvName), sep=';', encoding='UTF8', index_col=False)

        self.data = csv
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  

        if(os.getcwd().endswith("dataloader")):
            datasetDirectory = "../../Datasets"
        else:
            datasetDirectory = "../Datasets"
        dataset = 'OPERA7_wav'
        
        nomeFile = self.data.loc[idx, "NOME_FILE"]
        
        if(self.validation == "gender"):
            # Lettura della label associata all'emozione del file audio nel CSV
            label = self.data.loc[idx, "GENERE"]
        else: # validation = "emotion"
            # Lettura della label associata all'emozione del file audio nel CSV
            label = self.data.loc[idx, "EMOZIONE"]

        label = getLabel(label, self.validation)
       
        # Lettura del percorso del file audio dal CSV
        audio_file = datasetDirectory + "/{}".format(dataset) + nomeFile

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

        return {'image': pil, 'label': label, 'fileName': nomeFile}

if __name__ == "__main__":
    split = "val"
    validation = "emotion"
    gender = "all"

    operaTest = Opera7(gender=gender, validation=validation, split=split)
    
    print("Opera7 {} set successfully loaded".format(split))
    print("Loaded a total of {} samples".format(len(operaTest)))

    for i in range(5):
        random.seed(time.process_time())
        i = np.random.randint(len(operaTest))
        print("Label associata al file n. {}: {} ({}) - Nome file: {}".format(i, operaTest[i]['label'], getEmotion(operaTest[i]['label'], validation), operaTest[i]['fileName']))
        plotFigure(operaTest[i]['image'])
