import csv
import os
import tqdm
import pandas as pd

# Define any condition here
def countFemale(x):
  return x.startswith("f")

def countMale(x):
  return x.startswith("m")

def percentage(perc, totale):
  return round((perc * totale) / 100.0)

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

def makeEmovoCSV(dataset):
    
    Emovo_train = []
    
    Emovo_test = []
    
    countFemaleTrain = sum(countFemale(x) for x in dataset)
    countMaleTrain = sum(countMale(x) for x in dataset)
    
    percentageCountFemaleTrain = percentage(80, countFemaleTrain)
    
    percentageCountMaleTrain = percentage(80, countMaleTrain)
    
    c = 0
    k = 0
    
    for directory in dataset:
        if(directory.startswith("f")):
            if(c < percentageCountFemaleTrain):
                Emovo_train.append(directory)
            else:
                Emovo_test.append(directory)
            c += 1
        else:
            if(k < percentageCountMaleTrain):
                Emovo_train.append(directory)
            else:
                Emovo_test.append(directory)
            k += 1

    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('{}/train_emovo_female.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
        
    with open('{}/train_emovo_male.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
            
    with open('{}/val_emovo_female.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
        
    with open('{}/val_emovo_male.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
        
        print("Scrivo sul CSV di test per il dataset di Emovo ...")            
        for directory in Emovo_test:
            files = os.listdir(os.path.join(datasetsDirectory,"emovo/",directory))
            
            for wav_file in tqdm.tqdm(files):
                
                nomeFile = wav_file[0:3]
                
                genereFile = wav_file.split("-")[1]
                
                if(genereFile.startswith("f")):
                    genere = 'Donna'
                else:
                    genere = 'Uomo'
                
                emozione, valenza, arousal = getValues(nomeFile)
                     
                riga = ['/{}/'.format(genereFile) + wav_file, emozione, valenza, arousal, genere]
               
                if(directory.startswith("f")) :
                    genereCartella = "val_emovo_female"
                else:
                    genereCartella = "val_emovo_male"
                
                with open('{}/{}.csv'.format(datasetsDirectory,genereCartella), 'a', encoding='UTF8',newline='') as f:
                     writer = csv.writer(f,delimiter=';', quotechar='|', 
                                         quoting=csv.QUOTE_MINIMAL, 
                                         lineterminator="\n")
                     # scriviamo la riga
                     writer.writerow(riga)
                     f.close()

        print("\nScrivo sul CSV di train per il dataset di Emovo ...")                     
        for directory in Emovo_train:
            files = os.listdir(os.path.join(datasetsDirectory,"emovo/",directory))
            
            for wav_file in tqdm.tqdm(files):
                
                nomeFile = wav_file[0:3]
                
                genereFile = wav_file.split("-")[1]
                
                if(genereFile.startswith("f")):
                    genere = 'Donna'
                else:
                    genere = 'Uomo'
                
                emozione, valenza, arousal = getValues(nomeFile)
                     
                riga = ['/{}/'.format(genereFile) + wav_file, emozione, valenza, arousal, genere]
               
                if(directory.startswith("f")) :
                    genereCartella = "train_emovo_female"
                else:
                    genereCartella = "train_emovo_male"
                
                with open('{}/{}.csv'.format(datasetsDirectory,genereCartella), 'a', encoding='UTF8',newline='') as f:
                     writer = csv.writer(f,delimiter=';', quotechar='|', 
                                         quoting=csv.QUOTE_MINIMAL, 
                                         lineterminator="\n")
                     # scriviamo la riga
                     writer.writerow(riga)
                     f.close()

def makeWAV_Demos_CSV(dataset):
   
    wavdemos_female = []
    wavdemos_male = []
                            
    wavdemos_train = []
    wavdemos_test = []
    
    male = 0
    female = 0
    
    percentageCountFemale = percentage(80, 23)
    
    percentageCountMale = percentage(80, 45)
    
    for directory in dataset:
            files = os.listdir(os.path.join(datasetsDirectory, datasetDir, directory))
            for file in files:
                if (directory == "DEMOS"):
                    if (file[8:11] == "col"):
                        continue
                    
                    voceAttore = file.split("_")[1]
                    if (voceAttore == "f"):
                        wavdemos_female.append(file)
                    else:
                        wavdemos_male.append(file)
                else:
                    voceAttore = file.split("_")[0]
                    if (voceAttore == "f"):
                        wavdemos_female.append(file)
                    else:
                        wavdemos_male.append(file)
                        
    wavdemos_male_clone = wavdemos_male
    for file in wavdemos_male:
        if(file in wavdemos_male_clone):
            if(file.startswith("PR") or file.startswith("NP")):
                speaker = file.split("_")[2]
                p = list(filter(lambda x: (x.split("_")[2] == speaker or x.split("_")[1] == speaker), wavdemos_male))
                wavdemos_male_clone = list(set(wavdemos_male_clone) - set(p))
                if(male < percentageCountMale):
                    wavdemos_train.extend(p)
                else:
                    wavdemos_test.extend(p)
                male += 1
 
    wavdemos_female_clone = wavdemos_female
    for file in wavdemos_female:
        if(file in wavdemos_female_clone):
            if(file.startswith("PR") or file.startswith("NP")):
                speaker = file.split("_")[2]
                p = list(filter(lambda x: (x.split("_")[2] == speaker or x.split("_")[1] == speaker), wavdemos_female))
                wavdemos_female_clone = list(set(wavdemos_female_clone) - set(p))
                if(female < percentageCountFemale):
                    wavdemos_train.extend(p)
                else:
                    wavdemos_test.extend(p)
                female += 1

    wavdemos_train.sort()
    wavdemos_test.sort()
    
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('{}/train_demos_female.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
        
    with open('{}/train_demos_male.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
            
    with open('{}/val_demos_female.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
        
    with open('{}/val_demos_male.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()

    print("------------------------------")
    print("Scrivo sul CSV di train per il dataset di WAV_Demos ...")            
    for wav_file in tqdm.tqdm(wavdemos_train):
        
        if(wav_file.startswith("NP") or wav_file.startswith("PR")):
            voceAttore = wav_file.split("_")[1] 
            nomeFile = wav_file[8:11]
            cartella = "DEMOS"
        else:
            voceAttore = wav_file.split("_")[0]
            nomeFile = wav_file[5:8]
            cartella = "NEU"
        
        if(voceAttore == 'f'):
            genere = 'Donna'
            genereCartella = "train_demos_female"
        else:
            genere = 'Uomo'
            genereCartella = "train_demos_male"
        
        emozione, valenza, arousal = getValues(nomeFile)
                           
        header = ['/{}/'.format(cartella) + wav_file, emozione, valenza, arousal, genere]
                   
        with open('{}/{}.csv'.format(datasetsDirectory, genereCartella), 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                      
            # scriviamo la riga
            writer.writerow(header)  
            f.close()

    print("\nScrivo sul CSV di test per il dataset di WAV_Demos ...")
    for wav_file in tqdm.tqdm(wavdemos_test):
        
        if(wav_file.startswith("NP") or wav_file.startswith("PR")):
            voceAttore = wav_file.split("_")[1] 
            nomeFile = wav_file[8:11]
            cartella = "DEMOS"
        else:
            voceAttore = wav_file.split("_")[0]
            nomeFile = wav_file[5:8]
            cartella = "NEU"
        
        if(voceAttore == 'f'):
            genere = 'Donna'
            genereCartella = "val_demos_female"
        else:
            genere = 'Uomo'
            genereCartella = "val_demos_male"
        
        emozione, valenza, arousal = getValues(nomeFile)
                           
        header = ['/{}/'.format(cartella) + wav_file, emozione, valenza, arousal, genere]
                   
        with open('{}/{}.csv'.format(datasetsDirectory, genereCartella), 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                      
            # scriviamo la riga
            writer.writerow(header)  
            f.close()

def makeAllTrainCSV():

    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('{}/all_train_female.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()

    with open('{}/all_train_male.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
 
    with open('{}/all_train.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
    
    a = pd.read_csv("{}/train_demos_female.csv".format(datasetsDirectory))
    b = pd.read_csv("{}/train_emovo_female.csv".format(datasetsDirectory))
    result = pd.concat([a, b])

    result.to_csv("{}/all_train_female.csv".format(datasetsDirectory), index=False)
    
    # -------------------- #
    
    a = pd.read_csv("{}/train_demos_male.csv".format(datasetsDirectory))
    b = pd.read_csv("{}/train_emovo_male.csv".format(datasetsDirectory))
    result = pd.concat([a, b])

    result.to_csv("{}/all_train_male.csv".format(datasetsDirectory), index=False)

    # -------------------- #

    a = pd.read_csv("{}/all_train_male.csv".format(datasetsDirectory))
    b = pd.read_csv("{}/all_train_female.csv".format(datasetsDirectory))
    result = pd.concat([a, b])

    result.to_csv("{}/all_train.csv".format(datasetsDirectory), index=False)
    return

def makeAllTestCSV():
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('{}/all_val_female.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
        
    with open('{}/all_val_male.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()

    with open('{}/all_val.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()

    a = pd.read_csv("{}/val_demos_female.csv".format(datasetsDirectory))
    b = pd.read_csv("{}/val_emovo_female.csv".format(datasetsDirectory))
    result = pd.concat([a, b])

    result.to_csv("{}/all_val_female.csv".format(datasetsDirectory), index=False)
    
    # -------------------- #
    
    a = pd.read_csv("{}/val_demos_male.csv".format(datasetsDirectory))
    b = pd.read_csv("{}/val_emovo_male.csv".format(datasetsDirectory))
    result = pd.concat([a, b])

    result.to_csv("{}/all_val_male.csv".format(datasetsDirectory), index=False)

    # -------------------- #

    a = pd.read_csv("{}/all_val_male.csv".format(datasetsDirectory))
    b = pd.read_csv("{}/all_val_female.csv".format(datasetsDirectory))
    result = pd.concat([a, b])

    result.to_csv("{}/all_val.csv".format(datasetsDirectory), index=False)
    return

# ---------------------- MAIN ---------------------- #

AUDIO_FILE_EXTENSION = '.wav'

trainReader = []
testReader = []
trainingEmovo = []
testEmovo = []
trainingWAVDemos = []
testWAVDemos = []
emovo = []
WAVDemos = []
# --------------------

datasetsDirectory = 'Datasets/'

folders = os.listdir(datasetsDirectory)

# Prima puliamo i file CSV, poi li ricreiamo ogni volta
# in modo tale da avere sempre file puliti
for file in folders:
    if (file.endswith(".csv")):
        os.remove(os.path.join(datasetsDirectory, file))

folders = list(filter(lambda x: not x.endswith(".csv") and not x == '15 Free Ambient Sound Effects' and not x.endswith("tmp"), folders))

for datasetDir in folders:
    if(datasetDir.endswith(".csv")):
        continue
    datasetDirectories = os.listdir(os.path.join(datasetsDirectory, datasetDir))
    for singleDir in datasetDirectories:
        if(singleDir == "{}_augmentation".format(datasetDir)):
            continue
        for file in os.listdir(os.path.join(datasetsDirectory, datasetDir, singleDir)):
            if(file.endswith(".wav")):
                if (datasetDir == 'emovo'):  
                    emovo.append(singleDir)
                else:
                    WAVDemos.append(singleDir)
                break

# wav_Demos
makeWAV_Demos_CSV(WAVDemos)
print("\nCSV per WAV Demos Training & Test completato.")
print("------------------------------")

# Emovo
makeEmovoCSV(emovo)
print("\nCSV per Emovo Training & Test completato.")
print("------------------------------")

makeAllTrainCSV()
makeAllTestCSV()
print("\nCSV per train/test unificati completato.")
