import os
import shutil
import gdown

datasetsDirectory = 'Datasets/'

wav_demos_Dataset = 'wav_DEMoS'
emovo_Dataset = 'emovo'
ambient_Dataset = '15 Free Ambient Sound Effects'

wav_demos_URL = 'https://drive.google.com/uc?id=1O3b0Rua8_sNBmHlWF4nVgNU2lYOjsS__' 
emovo_URL = 'https://drive.google.com/uc?id=1AYv_1OROT-wL_OM7QsukQTz9eXJlNknu'
ambient_URL = 'https://drive.google.com/uc?id=1KrLixTDzI7j8V9k--aj4NL1UTAp13YoA'

# Se non esiste la cartella dei datasets, la creiamo
if not os.path.exists(datasetsDirectory):
    os.makedirs(datasetsDirectory)

# =============================================================================

# 15 Free Ambient Sound Effects
print("Downloading 15 Free Ambient Sound Effects dataset...")

gdown.download(ambient_URL, os.path.join(datasetsDirectory, ambient_Dataset + '.zip'))

print("Extracting 15 Free Ambient Sound Effects dataset...")
shutil.unpack_archive(os.path.join(datasetsDirectory, ambient_Dataset + ".zip"), os.path.join(datasetsDirectory, ambient_Dataset))
    
shutil.rmtree(os.path.join(datasetsDirectory, ambient_Dataset, "__MACOSX").replace('\\','/'))

source_folder = os.path.join(datasetsDirectory, ambient_Dataset, '{}/'.format(ambient_Dataset))
destination_folder = os.path.join(datasetsDirectory, '{}/'.format(ambient_Dataset))

for file_name in os.listdir(source_folder):
    # construct full file path
    source = source_folder + file_name
    destination = destination_folder + file_name
    
    # copy only files
    if os.path.isfile(source):
        if(file_name.endswith(".mp3")):
            shutil.move(source, destination)

for file_name in os.listdir(os.path.join(datasetsDirectory, ambient_Dataset, ambient_Dataset)):
    os.remove(os.path.join(datasetsDirectory, ambient_Dataset, ambient_Dataset, file_name))

os.rmdir(os.path.join(datasetsDirectory, ambient_Dataset, ambient_Dataset))

# =============================================================================

# wav_Demos
print("\nDownloading wav_DEMoS dataset...")

gdown.download(wav_demos_URL, './{}/{}.zip'.format(datasetsDirectory, wav_demos_Dataset), quiet=False)

print("Extracting wav_DEMoS dataset...")
shutil.unpack_archive(os.path.join(datasetsDirectory, wav_demos_Dataset + ".zip"), os.path.join(datasetsDirectory, wav_demos_Dataset))

# =============================================================================

# emovo
print("\nDownloading emovo dataset...")
 
gdown.download(emovo_URL, './{}/{}.zip'.format(datasetsDirectory, emovo_Dataset), quiet=False)

print("Extracting emovo dataset...")
shutil.unpack_archive(os.path.join(datasetsDirectory, emovo_Dataset + ".zip"), datasetsDirectory)

os.rename(os.path.join(datasetsDirectory,"EMOVO"), os.path.join(datasetsDirectory, emovo_Dataset))

# =============================================================================

# Cleaning files
print("\nCleaning files ...")

os.remove(os.path.join(datasetsDirectory, ambient_Dataset + '.zip'))
os.remove(os.path.join(datasetsDirectory, wav_demos_Dataset + ".zip"))
os.remove(os.path.join(datasetsDirectory, emovo_Dataset + ".zip"))

for file in os.listdir(datasetsDirectory):
    if(file.endswith("tmp")):
        os.remove(os.path.join(datasetsDirectory, file))
