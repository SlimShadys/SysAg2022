import csv
import os
import shutil

def makeEmovoValidationCSV():
    
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('CSV/emovo_Test.csv', 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
               
        testDataDirectory = os.path.join(datasetsDirectory, 'emovo') + '/test_data/'
        dirs = os.listdir(testDataDirectory)
        
        for singleDir in dirs:
            files = os.listdir(os.path.join(testDataDirectory,singleDir))
            
            if(singleDir == 'f1' or singleDir == 'f2' or singleDir == 'f3'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
            for wav_file in files:
                
                if (wav_file[0:3] == 'tri'):
                   emozione = 'Tristezza'
                   valenza = "Negativa"
                   arousal = "Bassa" 
                else:
                   emozione = 'Sorpresa'
                   valenza = "Positiva"
                   arousal = "Media"
                     
                riga = [wav_file, emozione, valenza, arousal, genere]
               
                with open('CSV/emovo_Test.csv', 'a', encoding='UTF8',newline='') as f:
                     writer = csv.writer(f,delimiter=';', quotechar='|', 
                                         quoting=csv.QUOTE_MINIMAL, 
                                         lineterminator="\n")
                     # scriviamo la riga
                     writer.writerow(riga)   
                     f.close()

def makeEmovoTrainingCSV():
    
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('CSV/emovo_Training.csv', 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
               
        trainingDataDirectory = os.path.join(resultDir, 'emovo')

        files = os.listdir(trainingDataDirectory)
            
        for wav_file in files:
            
            nomeFile = wav_file[0:3]
            
            voceAttore = wav_file.split("-")[1]
            
            if(voceAttore == 'f1' or voceAttore == 'f2' or voceAttore == 'f3'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
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
                 
            riga = [wav_file, emozione, valenza, arousal, genere]
           
            with open('CSV/emovo_Training.csv', 'a', encoding='UTF8',newline='') as f:
                 writer = csv.writer(f,delimiter=';', quotechar='|', 
                                     quoting=csv.QUOTE_MINIMAL, 
                                     lineterminator="\n")
                 # scriviamo la riga
                 writer.writerow(riga)
                 f.close()

def makeWAV_Demos_ValidationCSV():
    
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('CSV/wavDemos_Test.csv', 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
        
        testDataDirectory = os.path.join(datasetsDirectory, 'wav_DEMoS') + '/test_data/'
        
        dirs = os.listdir(testDataDirectory)
        
        # Es. Datasets/wav_Demos/test_data/DEMOS
        for singleDir in dirs:
            
            files = os.listdir(os.path.join(testDataDirectory,singleDir))
                
            for wav_file in files:
                
                if(singleDir == 'DEMOS'):
                    voceAttore = wav_file.split("_")[1]
                else:
                    voceAttore = wav_file.split("_")[0]
                
                if(voceAttore == 'f'):
                    genere = 'Donna'
                else:
                    genere = 'Uomo'
                
                nomeFile = wav_file[8:11]

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
                                   
                header = [wav_file, emozione, valenza, arousal, genere]
                           
                with open('CSV/wavDemos_Test.csv', 'a', encoding='UTF8',newline='') as f:
                    writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                              
                    # scriviamo la riga
                    writer.writerow(header)  
                    f.close()

def makeWAV_Demos_TrainingCSV():
    
    # nome_file, emozione, valenza, arousal
    header = ['NOME_FILE', 'EMOZIONE', 'VALENZA', 'AROUSAL', 'GENERE']
    
    with open('CSV/wavDemos_Training.csv', 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', 
                            quoting=csv.QUOTE_MINIMAL, 
                            lineterminator="\n")
        
        # scriviamo la riga
        writer.writerow(header)
        f.close()
        
        trainingDataDirectory = os.path.join(resultDir, 'wav_DEMoS')
        
        files = os.listdir(trainingDataDirectory)
            
        for wav_file in files:
            
            voceAttore = wav_file.split("_")[1]
            
            if(voceAttore == 'f'):
                genere = 'Donna'
            else:
                genere = 'Uomo'
            
            nomeFile = wav_file[8:11]

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
                               
            header = [wav_file, emozione, valenza, arousal, genere]
                       
            with open('CSV/wavDemos_Training.csv', 'a', encoding='UTF8',newline='') as f:
                writer = csv.writer(f,delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                          
                # scriviamo la riga
                writer.writerow(header)  
                f.close()

# ---------------------- MAIN ---------------------- #

CSVDirectory = 'CSV/'
datasetsDirectory = 'Datasets/'
resultDir = 'RESULT_DIR/'

# Prima puliamo la cartella dei CSV, poi la ricreiamo ogni volta
# in modo tale da avere sempre una cartella pulita
if(os.path.exists(CSVDirectory)):
    shutil.rmtree(CSVDirectory)
os.makedirs(CSVDirectory)    

folders = os.listdir(datasetsDirectory)

# Non ci interessa creare CSV sui suoni relativi ai rumori
folders.remove('15 Free Ambient Sound Effects')

# Emovo
makeEmovoValidationCSV()
print("CSV per Emovo Test completato.")

makeEmovoTrainingCSV()
print("CSV per Emovo Training completato.")

# wav_Demos
makeWAV_Demos_ValidationCSV()
print("CSV per WAV Demos Test completato.")

makeWAV_Demos_TrainingCSV()
print("CSV per WAV Demos Training completato.")
