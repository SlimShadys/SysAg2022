# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:09:17 2021

@author: PC
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:37:09 2021

@author: PC
"""

import sklearn
import librosa 
import numpy as np
import os
import IPython
from sklearn.neural_network import MLPClassifier
from estrazione_features import Features
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from emozionedafile import EmotionFromFilename
from joblib import load
import warnings
import wave
import contextlib
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from PIL import Image
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from random import shuffle
import cv2 
import matplotlib.pyplot as plt
from matplotlib import cm
from  tensorflow.keras.utils import to_categorical
import pickle
import librosa.display
import pylab
import IPython.display as ipd
from tqdm import tqdm

def timeshifter (y,sr):
    start_ = int(np.random.uniform(-4800,4800))
    print('time shift: ',start_)
    if start_ >= 0:
        wav_time_shift = np.r_[y[start_:], np.random.uniform(-0.001,0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), y[:start_]]
        ipd.Audio(wav_time_shift, rate=16000)
        M = librosa.feature.melspectrogram(wav_time_shift, sr, 
           fmax = sr/2, # Maximum frequency to be used on the on the MEL scale
           n_fft=2048, 
           hop_length=512, 
                                                   n_mels = 96, # As per the Google Large-scale audio CNN paper
                                                   power = 2) # Power = 2 refers to squared amplitude
                
                # Power in DB
    log_power = librosa.power_to_db(M, ref=np.max)# Covert to dB (log) scale
    pylab.figure(figsize=(3,3))
    pylab.axis('off') 
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(log_power, cmap=cm.jet)
    #pylab.savefig(IMG_DIR + f[:-4]+'-1.jpg', bbox_inches=None, pad_inches=0)
    #pylab.close()

def speedchange(y,sr):
    speed_rate = np.random.uniform(0.7,1.3)
    wav_speed_tune = cv2.resize(y, (1, int(len(y) * speed_rate))).squeeze()
    if len(wav_speed_tune) < y.shape[0]:
        pad_len = y.shape[0] - len(wav_speed_tune) 
        wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                               wav_speed_tune,
                               np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
    else: 
        cut_len = len(wav_speed_tune) - y.shape[0]
        wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+y.shape[0]]
    print('wav length: ', wav_speed_tune.shape[0])
    ipd.Audio(wav_speed_tune, rate=sr)
    M = librosa.feature.melspectrogram(wav_speed_tune, sr, 
       fmax = sr/2, # Maximum frequency to be used on the on the MEL scale
       n_fft=2048, 
       hop_length=512, 
                                               n_mels = 96, # As per the Google Large-scale audio CNN paper
                                               power = 2) # Power = 2 refers to squared amplitude
            
            # Power in DB
    log_power = librosa.power_to_db(M, ref=np.max)# Covert to dB (log) scale
    pylab.figure(figsize=(3,3))
    pylab.axis('off') 
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(log_power, cmap=cm.jet)
    #pylab.savefig(IMG_DIR + f[:-4]+'-2.jpg', bbox_inches=None, pad_inches=0)
    #pylab.close()

def noiseback(y,sr):
    bg_files = os.listdir('D:/Users/PC/Downloads/15-Free-Ambient-Sound-Effects/15 Free Ambient Sound Effects')
    chosen_bg_file = bg_files[np.random.randint(6)]
    bg, sr = librosa.load('D:/Users/PC/Downloads/15-Free-Ambient-Sound-Effects/15 Free Ambient Sound Effects/'+chosen_bg_file, sr=16000)
    print(chosen_bg_file,'|', bg.shape[0], bg.max(), bg.min())
    ipd.Audio(bg, rate=sr) # !! be prepared when playing the noise, bacause it's so ANNOYING !!
    start_ = np.random.randint(bg.shape[0]-y.shape[0])
    bg_slice = bg[start_ : start_+ y.shape[0]]
    wav_with_bg = y * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.1)
    ipd.Audio(wav_with_bg, rate=sr) 
    
    M = librosa.feature.melspectrogram(wav_with_bg, sr, 
       fmax = sr/2, # Maximum frequency to be used on the on the MEL scale
       n_fft=2048, 
       hop_length=512, 
                                               n_mels = 96, # As per the Google Large-scale audio CNN paper
                                               power = 2) # Power = 2 refers to squared amplitude
            
            # Power in DB
    log_power = librosa.power_to_db(M, ref=np.max)# Covert to dB (log) scale
    pylab.figure(figsize=(3,3))
    pylab.axis('off') 
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(log_power, cmap=cm.jet)
    #pylab.savefig(IMG_DIR + f[:-4]+'-3.jpg', bbox_inches=None, pad_inches=0)
    #pylab.close()

def alltransformations(y,sr):
    # time shift
    start_ = int(np.random.uniform(-4800,4800))
    print('time shift: ',start_)
    if start_ >= 0:
        wav_time_shift = np.r_[y[start_:], np.random.uniform(-0.001,0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), y[:start_]]
    ipd.Audio(wav_time_shift, rate=16000)
    
    
    #aumento velocità
    speed_rate = np.random.uniform(0.7,1.3)
    wav_speed_tune = cv2.resize(wav_time_shift, (1, int(len(wav_time_shift) * speed_rate))).squeeze()
    print('speed rate: %.3f' % speed_rate, '(lower is faster)')
    '''if len(wav_speed_tune) < 16000:
        pad_len = len(wav_speed_tune) 
        wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                               wav_speed_tune,
                               np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
    else: 
        cut_len = len(wav_speed_tune) - 16000
        wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+16000]'''
    if len(wav_speed_tune) < wav_time_shift.shape[0]:
        pad_len = wav_time_shift.shape[0] - len(wav_speed_tune) 
        wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                               wav_speed_tune,
                               np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
    else: 
        cut_len = len(wav_speed_tune) - wav_time_shift.shape[0]
        wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+wav_time_shift.shape[0]]
    print('wav length: ', wav_speed_tune.shape[0])
    ipd.Audio(wav_speed_tune, rate=sr)
    
    
    #rumori di sottofondo
    bg_files = os.listdir('D:/Users/PC/Downloads/15-Free-Ambient-Sound-Effects/15 Free Ambient Sound Effects')
    chosen_bg_file = bg_files[np.random.randint(6)]
    bg, sr = librosa.load('D:/Users/PC/Downloads/15-Free-Ambient-Sound-Effects/15 Free Ambient Sound Effects/'+chosen_bg_file, sr=16000)
    print(chosen_bg_file,'|', bg.shape[0], bg.max(), bg.min())
    ipd.Audio(bg, rate=sr) # !! be prepared when playing the noise, bacause it's so ANNOYING !!
    start_ = np.random.randint(bg.shape[0]-y.shape[0])
    bg_slice = bg[start_ : start_+ wav_speed_tune.shape[0]]
    wav_with_bg = wav_speed_tune * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.1)
    ipd.Audio(wav_with_bg, rate=sr) 
    
    M = librosa.feature.melspectrogram(wav_with_bg, sr, 
       fmax = sr/2, # Maximum frequency to be used on the on the MEL scale
       n_fft=2048, 
       hop_length=512, 
                                               n_mels = 96, # As per the Google Large-scale audio CNN paper
                                               power = 2) # Power = 2 refers to squared amplitude
            
            # Power in DB
    log_power = librosa.power_to_db(M, ref=np.max)# Covert to dB (log) scale
    return log_power




WAV_DIR = 'D:/Users/PC/Desktop/INFORMATICA 5Ci/Università materiale/TIROCINIO E TESI/Dataset/TEST/Film ecc/sistemati/'
IMG_DIR = 'D:/Users/PC/Desktop/INFORMATICA 5Ci/Università materiale/TIROCINIO E TESI/Dataset/TEST/Film ecc/spect'
wav_files = os.listdir(WAV_DIR)
log_power = []
c = 0
for f in tqdm(wav_files):
    try:
        c = c+1
        if(f[8:11] == 'col'):
            continue
        if(c < 3831):
            continue
        # Read wav-file
        y, sr = librosa.load(WAV_DIR+f, sr = 16000) # Use the default sampling rate of 22,050 Hz
        
        # Pre-emphasis filter
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        # Compute spectrogram
        M = librosa.feature.melspectrogram(y, sr, 
                                           fmax = sr/2, # Maximum frequency to be used on the on the MEL scale
                                           n_fft=2048, 
                                           hop_length=512, 
                                           n_mels = 96, # As per the Google Large-scale audio CNN paper
                                           power = 2) # Power = 2 refers to squared amplitude
        
        # Power in DB
        log_power = librosa.power_to_db(M, ref=np.max)# Covert to dB (log) scale
        
        # Plotting the spectrogram and save as JPG without axes (just the image)
        pylab.figure(figsize=(3,3))
        pylab.axis('off') 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(log_power, cmap=cm.jet)
        pylab.savefig(IMG_DIR + f[:-4]+'.jpg', bbox_inches=None, pad_inches=0)
        pylab.close()
        
        #aumento dati in mase a cambiamenti
        '''timeshifter(y, sr)
        speedchange(y, sr)
        noiseback(y, sr)'''
        
        log_power = alltransformations(y, sr)
        pylab.figure(figsize=(3,3))
        pylab.axis('off') 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(log_power, cmap=cm.jet)
        pylab.savefig(IMG_DIR + f[:-4] +'-1.jpg', bbox_inches=None, pad_inches=0)
        pylab.close()
        print("andato1")
        log_power = alltransformations(y, sr)
        pylab.figure(figsize=(3,3))
        pylab.axis('off') 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(log_power, cmap=cm.jet)
        pylab.savefig(IMG_DIR + f[:-4] +'-2.jpg', bbox_inches=None, pad_inches=0)
        pylab.close()
        print("andato2")
        '''log_power = alltransformations(y, sr)
        pylab.figure(figsize=(3,3))
        pylab.axis('off') 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(log_power, cmap=cm.jet)
        pylab.savefig(IMG_DIR + f[:-4] +'-3.jpg', bbox_inches=None, pad_inches=0)
        pylab.close()
        print("andato3")'''
        '''log_power = alltransformations(y, sr)
        pylab.figure(figsize=(3,3))
        pylab.axis('off') 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(log_power, cmap=cm.jet)
        pylab.savefig(IMG_DIR + f[:-4] +'-4.jpg', bbox_inches=None, pad_inches=0)
        pylab.close()
        print("andato4")'''
        
    except Exception as e:
        print(f, e)
        pass
    
