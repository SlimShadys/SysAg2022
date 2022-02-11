import os
import random
import time
import cv2
import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
import pylab
import soundfile as sf
import pandas as pd


from matplotlib import cm
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

def alltransformations(y, sr, fileName, i, training_dir, dataFrame):
    # time shift
    start_ = int(np.random.uniform(-4800,4800))
    #print('time shift: ',start_)
    if start_ >= 0:
        wav_time_shift = np.r_[y[start_:], np.random.uniform(-0.001,0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), y[:start_]]
    ipd.Audio(wav_time_shift, rate=16000)
                 
    #aumento velocità
    speed_rate = np.random.uniform(0.7,1.3)
    wav_speed_tune = cv2.resize(wav_time_shift, (1, int(len(wav_time_shift) * speed_rate))).squeeze()
    #print('speed rate: %.3f' % speed_rate, '(lower is faster)')
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
    #print('wav length: ', wav_speed_tune.shape[0])
    ipd.Audio(wav_speed_tune, rate=sr)
    
    #rumori di sottofondo
    randomBG = np.random.randint(len(dataFrame))
    bg = dataFrame.at[randomBG,'bg']
    sr = dataFrame.at[randomBG,'sr']
    #print(chosen_bg_file,'|', bg.shape[0], bg.max(), bg.min())
    ipd.Audio(bg, rate=sr) # !! be prepared when playing the noise, bacause it's so ANNOYING !!
    start_ = np.random.randint(bg.shape[0]-y.shape[0])
    bg_slice = bg[start_ : start_+ wav_speed_tune.shape[0]]
    
    #wav_with_bg = wav_speed_tune * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.1)
    
    wav_with_bg = bg_slice * np.random.uniform(0, 0.1)
    ipd.Audio(wav_with_bg, rate=sr) 
    
    finalArray = np.add(wav_speed_tune, wav_with_bg)
    
    newName = fileName.split("/")[2].split(".")[0] + "_" + str(i) + ".wav"
    
    # Ci salviamo l'audio modificato
    sf.write(os.path.join(training_dir, newName).replace('\\','/'), finalArray, sr)
    
    #M = librosa.feature.melspectrogram(wav_with_bg, sr, 
    #   fmax = sr/2, # Maximum frequency to be used on the on the MEL scale
    #   n_fft=2048, 
    #   hop_length=512, 
    #   n_mels = 96, # As per the Google Large-scale audio CNN paper
    #   power = 2) # Power = 2 refers to squared amplitude
            
    # Power in DB
    #log_power = librosa.power_to_db(M, ref=np.max)# Covert to dB (log) scale
    #return log_power
    return

def computeTransformation(wav_files):
    log_power = []

    # Pre-load ambient sounds
    print('-------------------------')    
    print("Pre-carico i suoni di sottofondo...")
    bg_files = os.listdir('Datasets/15 Free Ambient Sound Effects/')
    df = pd.DataFrame(columns=['bg','sr'])
    
    for bg_file in tqdm(bg_files):
        bg, sr = librosa.load('Datasets/15 Free Ambient Sound Effects/'+bg_file, sr=16000)
        df = df.append({
            "bg" : bg,
            "sr" : sr
            }, ignore_index=True)

    print('-------------------------')
    print("Creo i file di data augmentation..")
    for f in tqdm(wav_files):
        try:
            
            dataSet = f.split("/")[0]
            training_dir = os.path.join(datasetsDirectory, dataSet, "{}_augmentation".format(dataSet)).replace('\\','/')
            
            # Impostiamo un random seed ogni volta
            random.seed(time.process_time())

            # Read wav-file
            y, sr = librosa.load(os.path.join(datasetsDirectory,f), sr = 16000) # Use the default sampling rate of 22,050 Hz
            
            # Pre-emphasis filter
            pre_emphasis = 0.97
            y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
            
            # Compute spectrogram
            #M = librosa.feature.melspectrogram(y, sr, 
                                               #fmax = sr/2, # Maximum frequency to be used on the on the MEL scale
                                               #n_fft=2048, 
                                               #hop_length=512, 
                                               #n_mels = 96, # As per the Google Large-scale audio CNN paper
                                               #power = 2) # Power = 2 refers to squared amplitude
            
            # Power in DB
            #log_power = librosa.power_to_db(M, ref=np.max)# Covert to dB (log) scale
            
            # Plotting the spectrogram and save as JPG without axes (just the image)
            #pylab.figure(figsize=(3,3))
            #pylab.axis('off') 
            #pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
            #pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
            #librosa.display.specshow(log_power, cmap=cm.jet)
            #pylab.savefig(IMG_DIR + f[:-4]+'.jpg', bbox_inches=None, pad_inches=0)
            #pylab.close()
            
            #aumento dati in mase a cambiamenti
            '''timeshifter(y, sr)
            speedchange(y, sr)
            noiseback(y, sr)'''
            
            i = 0
            
            # Implementa le varie trasformazioni per due volte
            # Ogni volta però, facciamo trasformazioni random
            log_power = alltransformations(y, sr, f, i, training_dir, df)
            #pylab.figure(figsize=(3,3))
            #pylab.axis('off') 
            #pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
            #librosa.display.specshow(log_power, cmap=cm.jet)
            #pylab.savefig(IMG_DIR + f[:-4] +'-1.jpg', bbox_inches=None, pad_inches=0)
            #pylab.close()

            i += 1

            # Impostiamo un random seed ogni volta
            random.seed(time.process_time())
            
            log_power = alltransformations(y, sr, f, i, training_dir, df)
            #pylab.figure(figsize=(3,3))
            #pylab.axis('off') 
            #pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
            #librosa.display.specshow(log_power, cmap=cm.jet)
            #pylab.savefig(IMG_DIR + f[:-4] +'-2.jpg', bbox_inches=None, pad_inches=0)
            #pylab.close()
            
        except Exception as e:
            print(f, e)
            pass

# ---------------------- MAIN ---------------------- #

datasetsDirectory = 'Datasets/'

# Se non esiste la cartella dei risultati, la creiamo
if not os.path.exists(datasetsDirectory):
    print("La cartella relativa ai Datasets non esiste!")
    print("Creare la cartella ''Datasets'' e inserirci i dataset necessari.")
    exit(0)

folders = os.listdir(datasetsDirectory)

folders = list(filter(lambda x: not x.endswith(".csv") and not x == '15 Free Ambient Sound Effects', folders))

for folder in folders:
    
    dataAugmentation = "{}_augmentation".format(folder)
    
    if not(os.path.exists(os.path.join(datasetsDirectory,folder,dataAugmentation))):
           os.makedirs(os.path.join(datasetsDirectory,folder,dataAugmentation))
    
columns = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
df = pd.read_csv('{}/train.csv'.format(datasetsDirectory), sep=";", usecols=columns)
wav_files = df['NOME_FILE'].tolist() #you can also use df['column_name']

if(len(wav_files) > 0):
    
    print('-------------------------')
    print(F"Trasformo la cartella: {os.path.join(datasetsDirectory, folder)}")
    computeTransformation(wav_files)
    print('\n')
else:
    print("Sei sicuro che esista un CSV di training in: {datasetsDirectory}?")
    