import os
import shutil
import tqdm

def percentage(perc, totale):
  return round((perc * totale) / 100.0)

# ---------------------- MAIN ---------------------- #

AUDIO_FILE_EXTENSION = '.wav'
datasetsDirectory = 'Datasets/'
folders = os.listdir(datasetsDirectory)

# Non ci interessa effettuare splitting sui suoni relativi ai rumori
folders.remove('15 Free Ambient Sound Effects')

for folder in folders:
        
    print(F'\nCreo lo splitting per la cartella {folder}...')
    print('-------------------------')
    
    testDataDirectory = os.path.join(datasetsDirectory, folder) + '/test_data/'
    trainingDataDirectory = os.path.join(datasetsDirectory, folder) + '/training_data/'

    # Se non esistono queste cartelle, le creo
    if not os.path.exists(testDataDirectory):
        os.makedirs(testDataDirectory)
        
    if not os.path.exists(trainingDataDirectory):
        os.makedirs(trainingDataDirectory)
    
    # Filtriamo la lista delle cartelle
    # Se vogliamo eseguire lo splitting più di una volta, il compilatore
    # prenderà anche le cartelle "test_data" e "training_data", quindi
    # togliamole prima di eseguire il codice di splitting
    dirs = os.listdir(os.path.join(datasetsDirectory, folder))
    dirs = list(filter(lambda x: 'test_data' not in x and 'training_data' not in x, dirs))
    
    # Per ogni sottocartella
    #
    # Es. Datasets/emovo/f1
    for singleDir in dirs:
        
        # Controlliamo tutti i file e filtriamo solamente per i file che
        # ci interessano (.wav)
        files = os.listdir(os.path.join(datasetsDirectory, folder, singleDir))
        files = list(filter(lambda x: AUDIO_FILE_EXTENSION in x, files))
        
        # Può capitare che in una cartella non ci siano file audio, quindi
        # se la lunghezza è = 0, skippiamo
        if(len(files) > 0):
            numeroFile = len(files)
        else:
            continue
        
        
        # Se non esistono queste cartelle, le creo
        if not os.path.exists(trainingDataDirectory):
            os.makedirs(trainingDataDirectory)
            
        if not os.path.exists(testDataDirectory):
            os.makedirs(testDataDirectory)
        
        # Mi calcolo, in percentuale, il numero di file contenuti in ogni cartella
        # - 80% per il training
        # - 20% per il testing
        trainingPerc = percentage(80, numeroFile)
        testingPerc = percentage(20, numeroFile)
        
        # I primi 80% dei file, li trasferisco nella cartella relativa al training,
        # il resto (20%), li trasferisco nella cartella relativa al testing
        for i in tqdm.trange(trainingPerc):
            source = os.path.join(datasetsDirectory,folder,singleDir,files[i])
            dest = os.path.join(trainingDataDirectory,files[i]).replace('\\','/')
            shutil.copyfile(source, dest)

        # Quando passeremo ai file di test, aumentiamo la variabile i (relativa al)
        # training di 1 unità. In questo modo l'ultimo file di training, non verrà
        # ri-copiato nella cartella di testing
        i += 1

        for j in tqdm.trange(testingPerc):
            source = os.path.join(datasetsDirectory,folder,singleDir,files[i+j])
            dest = os.path.join(testDataDirectory,files[i+j])
            shutil.copyfile(source, dest)
