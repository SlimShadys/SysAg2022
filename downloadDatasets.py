import argparse
import os
import shutil

import gdown
from mega import Mega

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Configuration train phase")
parser.add_argument("-ud","--uses_drive", type=str2bool, const=True, nargs='?', default=True, help='Whether using Drive for downloading models. If False, rely on Mega')
parser.add_argument("-ad","--augmentation_downloaded", type=str2bool, const=True, nargs='?', default=True, help='Whether augmentation files should be downloaded')
args = parser.parse_args()

augmentationDownloaded = args.augmentation_downloaded
usesDrive = args.uses_drive

datasetsDirectory = 'Datasets/'

wav_demos_Dataset = 'wav_DEMoS'
emovo_Dataset = 'emovo'
opera7_Dataset = 'OPERA7_wav'
ambient_Dataset = '15 Free Ambient Sound Effects'

wav_demos_URL = 'https://drive.google.com/uc?id='         + '1O3b0Rua8_sNBmHlWF4nVgNU2lYOjsS__' + '&export=download&confirm=t' 
emovo_URL = 'https://drive.google.com/uc?id='             + '1AYv_1OROT-wL_OM7QsukQTz9eXJlNknu' + '&export=download&confirm=t'
opera7_URL = 'https://drive.google.com/uc?id='            + '1spBRz0-_0Hvy9AYEDdoODR3sBLtvHxvP' + '&export=download&confirm=t'
ambient_URL = 'https://drive.google.com/uc?id='           + '1KrLixTDzI7j8V9k--aj4NL1UTAp13YoA' + '&export=download&confirm=t'

datasets_URL_Drive = 'https://drive.google.com/uc?id='    + '1lXpEGNlyBm1OSFqyCDJf5UUmx9LPLiDQ' + '&export=download&confirm=t'
datasets_URL_Mega = 'https://mega.nz/file/'               + '8VFikTJJ#YpTUI6d7auf5e2fBDzGDohKYMeE5NQLsxGeHhTebmYs'

# =============================================================================

# Se non esiste la cartella dei datasets, la creiamo
if not os.path.exists(datasetsDirectory):
    os.makedirs(datasetsDirectory)

if(augmentationDownloaded):
    # Datasets
    print("\nDownloading datasets...")
    
    if (usesDrive):
        gdown.download(datasets_URL_Drive, 'Datasets.zip', quiet=False)
    else:
        # Mega API
        mega = Mega()
        m = mega.login()
        try:
            m.download_url(datasets_URL_Mega)
        except PermissionError:
            pass

    print("\nExtracting datasets...")
    shutil.unpack_archive("Datasets.zip", datasetsDirectory)

    # Cleaning files
    print("\nCleaning files ...")

    os.remove("Datasets.zip")

    for file in os.listdir(os.getcwd()):
        if(file.endswith("tmp")):
            os.remove(os.path.join(datasetsDirectory, file))
else:
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

    # Opera7
    print("\nDownloading Opera7 dataset...")
    
    gdown.download(opera7_URL, './{}/{}.zip'.format(datasetsDirectory, opera7_Dataset), quiet=False)

    print("Extracting Opera7 dataset...")
    shutil.unpack_archive(os.path.join(datasetsDirectory, opera7_Dataset + ".zip"), datasetsDirectory)

    # =============================================================================

    # Cleaning files
    print("\nCleaning files ...")

    os.remove(os.path.join(datasetsDirectory, ambient_Dataset + '.zip'))
    os.remove(os.path.join(datasetsDirectory, wav_demos_Dataset + ".zip"))
    os.remove(os.path.join(datasetsDirectory, emovo_Dataset + ".zip"))
    os.remove(os.path.join(datasetsDirectory, opera7_Dataset + ".zip"))

    for file in os.listdir(datasetsDirectory):
        if(file.endswith("tmp")):
            os.remove(os.path.join(datasetsDirectory, file))
