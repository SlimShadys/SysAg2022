import csv
import os
import tqdm

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

def makeEmovoValidationCSV(dataset):
    
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('{}/emovo_Test.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
                      
        for wav_file in tqdm.tqdm(dataset):
            
            genereFile = wav_file.split("-")[1]
            if(genereFile == 'f1' or genereFile == 'f2' or genereFile == 'f3'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
            nomeFile = wav_file[0:3]
            
            emozione, valenza, arousal = getValues(nomeFile)

            riga = [os.path.join('emovo') + '/{}/'.format(genereFile) + wav_file, emozione, valenza, arousal, genere]
           
            with open('{}/emovo_Test.csv'.format(datasetsDirectory), 'a', encoding='UTF8',newline='') as f:
                 writer = csv.writer(f,delimiter=';', quotechar='|', 
                                     quoting=csv.QUOTE_MINIMAL, 
                                     lineterminator="\n")
                 # scriviamo la riga
                 writer.writerow(riga)   
                 f.close()

def makeEmovoTrainingCSV(dataset):
    
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('{}/emovo_Training.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
            
        for wav_file in tqdm.tqdm(dataset):
            
            nomeFile = wav_file[0:3]
            
            genereFile = wav_file.split("-")[1]
            
            if(genereFile == 'f1' or genereFile == 'f2' or genereFile == 'f3'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
            emozione, valenza, arousal = getValues(nomeFile)
                 
            riga = [os.path.join('emovo') + '/{}/'.format(genereFile) + wav_file, emozione, valenza, arousal, genere]
           
            with open('{}/emovo_Training.csv'.format(datasetsDirectory), 'a', encoding='UTF8',newline='') as f:
                 writer = csv.writer(f,delimiter=';', quotechar='|', 
                                     quoting=csv.QUOTE_MINIMAL, 
                                     lineterminator="\n")
                 # scriviamo la riga
                 writer.writerow(riga)
                 f.close()

def makeWAV_Demos_ValidationCSV(dataset):
    
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('{}/wavDemos_Test.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()

        for wav_file in tqdm.tqdm(dataset):
            
            if(wav_file.startswith("NP") or wav_file.startswith("PR")):
                voceAttore = wav_file.split("_")[1]
                cartella = "DEMOS"
            else:
                voceAttore = wav_file.split("_")[0]
                cartella = "NEU"
                
            if(voceAttore == 'f'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
            nomeFile = wav_file[8:11]

            emozione, valenza, arousal = getValues(nomeFile)
                                           
            header = [os.path.join('wav_DEMoS') + '/{}/'.format(cartella) + wav_file, emozione, valenza, arousal, genere]
                       
            with open('{}/wavDemos_Test.csv'.format(datasetsDirectory), 'a', encoding='UTF8',newline='') as f:
                writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                          
                # scriviamo la riga
                writer.writerow(header)  
                f.close()

def makeWAV_Demos_TrainingCSV(dataset):
    
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('{}/wavDemos_Training.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
            
        for wav_file in tqdm.tqdm(dataset):
            
            if(wav_file.startswith("NP") or wav_file.startswith("PR")):
                voceAttore = wav_file.split("_")[1]
                cartella = "DEMOS"
            else:
                voceAttore = wav_file.split("_")[0]
                cartella = "NEU"
            
            if(voceAttore == 'f'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
            nomeFile = wav_file[8:11]

            emozione, valenza, arousal = getValues(nomeFile)
                               
            header = [os.path.join('wav_DEMoS') + '/{}/'.format(cartella) + wav_file, emozione, valenza, arousal, genere]
                       
            with open('{}/wavDemos_Training.csv'.format(datasetsDirectory), 'a', encoding='UTF8',newline='') as f:
                writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                          
                # scriviamo la riga
                writer.writerow(header)  
                f.close()

def makeAllTrainCSV(wav_demos, emovo):

    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('{}/train.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()   
        
    with open('{}/train.csv'.format(datasetsDirectory), 'a', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        
        for line in tqdm.tqdm(wav_demos):
            
            if(line.startswith("NP") or line.startswith("PR")):
                voceAttore = line.split("_")[1]
                cartella = "DEMOS"
            else:
                voceAttore = line.split("_")[0]
                cartella = "NEU"
            
            if(voceAttore == 'f'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
            nomeFile = line[8:11]

            emozione, valenza, arousal = getValues(nomeFile)
                               
            header = [os.path.join('wav_DEMoS') + '/{}/'.format(cartella) + line, emozione, valenza, arousal, genere]
            
            # scriviamo la riga
            writer.writerow(header)
            
        for line_2 in tqdm.tqdm(emovo):
            nomeFile = line_2[0:3]
            
            genereFile = line_2.split("-")[1]
            
            if(genereFile == 'f1' or genereFile == 'f2' or genereFile == 'f3'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
            emozione, valenza, arousal = getValues(nomeFile)
                 
            riga = [os.path.join('emovo') + '/{}/'.format(genereFile) + line_2, emozione, valenza, arousal, genere]
            
            # scriviamo la riga
            writer.writerow(riga)
            
        f.close()
        
    return

def makeAllTestCSV(wav_demos, emovo):
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('{}/validation.csv'.format(datasetsDirectory), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()     

    with open('{}/validation.csv'.format(datasetsDirectory), 'a', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        
        for line in tqdm.tqdm(wav_demos):
            
            if(line.startswith("NP") or line.startswith("PR")):
                voceAttore = line.split("_")[1]
                cartella = "DEMOS"
            else:
                voceAttore = line.split("_")[0]
                cartella = "NEU"
                
            if(voceAttore == 'f'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
            nomeFile = line[8:11]
    
            emozione, valenza, arousal = getValues(nomeFile)
                                           
            header = [os.path.join('wav_DEMoS') + '/{}/'.format(cartella) + line, emozione, valenza, arousal, genere]
            # scriviamo la riga
            writer.writerow(header)
            
        for line_2 in tqdm.tqdm(emovo):
            nomeFile = line_2[0:3]
            
            genereFile = line_2.split("-")[1]
            
            if(genereFile == 'f1' or genereFile == 'f2' or genereFile == 'f3'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
            emozione, valenza, arousal = getValues(nomeFile)
                 
            riga = [os.path.join('emovo') + '/{}/'.format(genereFile) + line_2, emozione, valenza, arousal, genere]
            writer.writerow(riga)
        
        f.close()
    
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
        folders.remove(file)
    if (file == '15 Free Ambient Sound Effects'):
        folders.remove(file)

folders = folders

for datasetDir in folders:
    if(datasetDir.endswith(".csv")):
        continue
    datasetDirectories = os.listdir(os.path.join(datasetsDirectory, datasetDir))
    for singleDir in datasetDirectories:
        files = os.listdir(os.path.join(datasetsDirectory, datasetDir, singleDir))
        for file in files:
            if (file.endswith(".wav")):
                if (datasetDir == 'emovo'):  
                    emovo.append(file)
                else:
                    WAVDemos.append(file)

trainingEmovoPerc = percentage(80, len(emovo))
trainingWAVPerc = percentage(80, len(WAVDemos))

testEmovo = emovo[trainingEmovoPerc:]
testWAVDemos = WAVDemos[trainingWAVPerc:]

print("Preparo i dataset..")
for i in tqdm.trange(trainingEmovoPerc):
    trainingEmovo.append(emovo[i])
  
for j in tqdm.trange(trainingWAVPerc):
    trainingWAVDemos.append(WAVDemos[j])    
print("------------------------------")

# wav_Demos
makeWAV_Demos_TrainingCSV(trainingWAVDemos)
print("CSV per WAV Demos Training completato.")
print("------------------------------")

makeWAV_Demos_ValidationCSV(testWAVDemos)
print("CSV per WAV Demos Test completato.")
print("------------------------------")

# Emovo
makeEmovoTrainingCSV(trainingEmovo)
print("CSV per Emovo Training completato.")
print("------------------------------")

makeEmovoValidationCSV(testEmovo)
print("CSV per Emovo Test completato.")
print("------------------------------")

makeAllTrainCSV(trainingWAVDemos, trainingEmovo)
makeAllTestCSV(testWAVDemos, testEmovo)
print("\nCSV per train/test unificati completato.")