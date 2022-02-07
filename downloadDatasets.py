import os
from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdl import download
import zipfile
import shutil

datasetsDirectory = 'Datasets/'

wav_demos_Dataset = 'wav_DEMoS'
emovo_Dataset = 'emovo'
ambient_Dataset = '15 Free Ambient Sound Effects'

# Se non esiste la cartella dei datasets, la creiamo
if not os.path.exists(datasetsDirectory):
    os.makedirs(datasetsDirectory)

# 15 Free Ambient Sound Effects
print("Downloading 15 Free Ambient Sound Effects dataset...")
url = "http://pbblogassets.s3.amazonaws.com/uploads/2016/09/15-Free-Ambient-Sound-Effects.zip"
download(url, os.path.join(datasetsDirectory, "15-Free-Ambient-Sound-Effects.zip"))
    
with zipfile.ZipFile(os.path.join(datasetsDirectory,"15-Free-Ambient-Sound-Effects.zip"), 'r') as zipFile:
    zipFile.extractall(os.path.join(datasetsDirectory, ambient_Dataset))
    
shutil.rmtree(os.path.join(datasetsDirectory, ambient_Dataset, "__MACOSX").replace('\\','/'))

source_folder = os.path.join(datasetsDirectory, ambient_Dataset, '15 Free Ambient Sound Effects/')
destination_folder = os.path.join(datasetsDirectory, "15 Free Ambient Sound Effects/")

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

os.remove(os.path.join(datasetsDirectory,"15-Free-Ambient-Sound-Effects.zip"))

# =============================================================================

# wav_Demos
print("\nDownloading wav_DEMoS dataset...")
gdd.download_file_from_google_drive(file_id='1eAhmyPW8A-O-_BkbjzKYUfta8Mks3h_k',
                                    dest_path='./Datasets/' + wav_demos_Dataset + '.zip',
                                    unzip=False)

with zipfile.ZipFile(os.path.join(datasetsDirectory, wav_demos_Dataset + ".zip"), 'r') as zipFile:
    zipFile.extractall(os.path.join(datasetsDirectory, wav_demos_Dataset))
    
os.remove(os.path.join(datasetsDirectory, wav_demos_Dataset + ".zip"))

# =============================================================================

# emovo
print("\nDownloading emovo dataset...")
gdd.download_file_from_google_drive(file_id='1SUtaKeA-LYnKaD3qv87Y5wYgihJiNJAo',
                                    dest_path='./Datasets/' + emovo_Dataset + '.zip',
                                    unzip=True)

os.rename(os.path.join(datasetsDirectory,"EMOVO"), os.path.join(datasetsDirectory, emovo_Dataset))
os.remove(os.path.join(datasetsDirectory, emovo_Dataset + ".zip"))
