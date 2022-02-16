import os
import random
import time
import cv2
import csv
import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import shutil
import pandas as pd

from tqdm import tqdm

def getValues(nomeFile):
    emozione = ""
    valenza = ""
    arousal = ""
    
    if (nomeFile == 'dis'):
        emozione = "Disgusto"
        valenza = "Negativa"
        arousal = "Bassa" 
    elif (nomeFile == 'gio') :
        emozione = "Gioia"
        valenza = "Positiva"
        arousal = "Alta"
    elif (nomeFile == 'pau'):
        emozione = "Paura"
        valenza = "Neutrale"
        arousal = "Alta"
    elif (nomeFile == 'rab'):
        emozione = "Rabbia"
        valenza = "Negativa"
        arousal = "Alta"
    elif (nomeFile == 'sor'):
        emozione = "Sorpresa"
        valenza = "Positiva"
        arousal = "Media"
    elif (nomeFile == 'tri'):
        emozione = "Tristezza"
        valenza = "Negativa"
        arousal = "Bassa"
    elif (nomeFile == 'col'):
        emozione = "Colpa"
        valenza = "Negativa"
        arousal = "Bassa"
    else :
       emozione = "Neutrale"
       valenza = "Neutrale"
       arousal = "Media"
    return emozione, valenza, arousal

def alltransformations(y, sr, fileName, i, training_dir, dataFrame):

    # ------ Time Shift------ #
    start_ = int(np.random.uniform(-4800,4800))
    if start_ >= 0:
        wav_time_shift = np.r_[y[start_:], np.random.uniform(-0.001,0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), y[:start_]]
    ipd.Audio(wav_time_shift, rate=16000)
                 
    # ------ Aumento velocità ------ #
    speed_rate = np.random.uniform(0.7,1.3)
    wav_speed_tune = cv2.resize(wav_time_shift, (1, int(len(wav_time_shift) * speed_rate))).squeeze()
    if len(wav_speed_tune) < wav_time_shift.shape[0]:
        pad_len = wav_time_shift.shape[0] - len(wav_speed_tune) 
        wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                               wav_speed_tune,
                               np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
    else: 
        cut_len = len(wav_speed_tune) - wav_time_shift.shape[0]
        wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+wav_time_shift.shape[0]]
    ipd.Audio(wav_speed_tune, rate=sr)
    
    # ------ Rumori di sottofondo ------ #
    randomBG = np.random.randint(len(dataFrame))
    bg = dataFrame.at[randomBG,'bg']
    sr = dataFrame.at[randomBG,'sr']
    ipd.Audio(bg, rate=sr) # !! be prepared when playing the noise, bacause it's so ANNOYING !!
    start_ = np.random.randint(bg.shape[0]-y.shape[0])
    bg_slice = bg[start_ : start_+ wav_speed_tune.shape[0]]
    
    wav_with_bg = bg_slice * np.random.uniform(0, 0.1)
    ipd.Audio(wav_with_bg, rate=sr) 
    
    finalArray = np.add(wav_speed_tune, wav_with_bg)
    
    newName = fileName.split("/")[2].split(".")[0] + "_" + str(i) + ".wav"
    
    # Ci salviamo l'audio modificato
    sf.write(os.path.join(training_dir, newName).replace('\\','/'), finalArray, sr)
        
    if(training_dir.split("/")[1] == 'emovo'):
        nomeFile = fileName.split("/")[2][0:3]
        
        genereFile = fileName.split("/")[2].split("-")[1]
        
        if(genereFile.startswith("f")):
            genere = 'Donna'
            genereCartella = "train_emovo_female"
            gender = 'female'
        else:
            genere = 'Uomo'
            genereCartella = "train_emovo_male"
            gender = 'male'
        
        emozione, valenza, arousal = getValues(nomeFile)
            
        riga = ['/{}/'.format(training_dir.split("/")[2]) + newName, emozione, valenza, arousal, genere]           
        
        with open('{}/{}.csv'.format(datasetsDirectory,genereCartella), 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f,delimiter=';', quotechar='|', 
                                quoting=csv.QUOTE_MINIMAL, 
                                lineterminator="\n")
            # scriviamo la riga
            writer.writerow(riga)
            f.close()

        with open('{}/train_emovo_all.csv'.format(datasetsDirectory), 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f,delimiter=';', quotechar='|', 
                                quoting=csv.QUOTE_MINIMAL, 
                                lineterminator="\n")
            # scriviamo la riga
            writer.writerow(riga)
            f.close()
           
        with open('{}/all_train_{}.csv'.format(datasetsDirectory, gender), 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                      
            # scriviamo la riga
            writer.writerow(riga)  
            f.close()
            
        with open('{}/all_train.csv'.format(datasetsDirectory), 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                      
            # scriviamo la riga
            writer.writerow(riga)
            f.close()
    else:
        if(fileName.split("/")[2].startswith("NP") or fileName.split("/")[2].startswith("PR")):
            voceAttore = fileName.split("/")[2].split("_")[1] 
            nomeFile = fileName.split("/")[2][8:11]
        else:
            voceAttore = fileName.split("/")[2].split("_")[0]
            nomeFile = fileName.split("/")[2][5:8]
        
        if(voceAttore == 'f'):
            genere = 'Donna'
            gender = 'female'
            genereCartella = "train_demos_female"
        else:
            genere = 'Uomo'
            gender = 'male'
            genereCartella = "train_demos_male"
        
        emozione, valenza, arousal = getValues(nomeFile)
                           
        header = ['/{}/'.format(training_dir.split("/")[2]) + newName, emozione, valenza, arousal, genere]
                   
        with open('{}/{}.csv'.format(datasetsDirectory, genereCartella), 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                      
            # scriviamo la riga
            writer.writerow(header)  
            f.close()

        with open('{}/train_demos_all.csv'.format(datasetsDirectory), 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                      
            # scriviamo la riga
            writer.writerow(header)  
            f.close()

        with open('{}/all_train_{}.csv'.format(datasetsDirectory, gender), 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                      
            # scriviamo la riga
            writer.writerow(header)  
            f.close()
            
        with open('{}/all_train.csv'.format(datasetsDirectory), 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                      
            # scriviamo la riga
            writer.writerow(header)  
            f.close()
    
    return

def sortCSV():
    
    folders = os.listdir(datasetsDirectory)
    
    folders = list(filter(lambda x: x.endswith(".csv"), folders))
    
    print('-------------------------')
    print("Ordino i vari CSV ...")
    
    for csvFile in folders:
        
        csvData = pd.read_csv("{}/{}".format(datasetsDirectory, csvFile), sep=';')
        
        csvData = csvData.reset_index(drop=True)
        idx = csvData['NOME_FILE'].str.split('/', expand=True).sort_values([2,1,0]).index
        csvData = csvData.reindex(idx).reset_index(drop=True)
        
        #csvData.sort_values(csvData.columns[0], 
        #                axis=0,
        #                inplace=True)
        
        csvData.to_csv("{}/{}".format(datasetsDirectory, csvFile), index=False, sep=";")

def computeTransformation(wav_files):

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
    
    wav_files.sort()
    
    for f in tqdm(wav_files):
        try:
            if(f.split("/")[1] == 'DEMOS' or f.split("/")[1] == 'NEU'):
                dataSet = 'wav_DEMoS'
            else:
                dataSet = 'emovo'
                            
            training_dir = os.path.join(datasetsDirectory, dataSet, "{}_augmentation".format(dataSet)).replace('\\','/')
            
            # Impostiamo un random seed ogni volta
            random.seed(time.process_time())

            # Read wav-file
            y, sr = librosa.load((datasetsDirectory + dataSet + f).replace('\\','/'), sr = 16000) # Use the default sampling rate of 22,050 Hz
            
            # Pre-emphasis filter
            pre_emphasis = 0.97
            y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
            
            i = 0
            
            # Implementa le varie trasformazioni per due volte
            # Ogni volta però, facciamo trasformazioni random
            alltransformations(y, sr, f, i, training_dir, df)

            i += 1

            # Impostiamo un random seed ogni volta
            random.seed(time.process_time())
            
            alltransformations(y, sr, f, i, training_dir, df)
            
        except Exception as e:
            print(f, e)
            pass
        


# ---------------------- MAIN ---------------------- #

emovo = []
WAVDemos = []

datasetsDirectory = 'Datasets/'

# Se non esiste la cartella dei risultati, la creiamo
if not os.path.exists(datasetsDirectory):
    print("La cartella relativa ai Datasets non esiste!")
    print("Creare la cartella ''Datasets'' e inserirci i dataset necessari.")
    print("- Eseguire ''python downloadDatasets.py''")
    exit(0)

folders = os.listdir(datasetsDirectory)

folders = list(filter(lambda x: not x.endswith(".csv") and not x == '15 Free Ambient Sound Effects' and not x.endswith("tmp"), folders))

print("Pulisco le cartelle di data augmentation ...")
for folder in folders:
    
    dataAugmentation = "{}_augmentation".format(folder)
    
    if os.path.exists(os.path.join(datasetsDirectory,folder,dataAugmentation)):
        shutil.rmtree((os.path.join(datasetsDirectory,folder,dataAugmentation)))
        
    os.makedirs(os.path.join(datasetsDirectory,folder,dataAugmentation))

columns = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
df = pd.read_csv('{}/all_train.csv'.format(datasetsDirectory), sep=";", usecols=columns)
wav_files = df['NOME_FILE'].tolist() #you can also use df['column_name']

if(len(wav_files) > 0):
    sortCSV()    
    print('-------------------------')
    print("Trasformo le cartelle: {}, {}".format(folders[0], folders[1]))
    computeTransformation(wav_files)
    print('-------------------------')
    print("Data augmentation completata!")
else:
    print("Sei sicuro che esista un CSV di training in: {datasetsDirectory}?")
    