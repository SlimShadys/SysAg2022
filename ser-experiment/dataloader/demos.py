import torch.utils.data as data
import pandas as pd
from PIL import Image


class Demos(data.Dataset):

    def __init__(self, gender, split='train', transform=None):
        self.transform = transform
        self.split = split
        self.gender = gender

        # TODO: implementare il caricamento del CSV e settare la cartella dove sono presenti i file audio in self.audios
        if self.split == "train":
            if(gender == 'f'):
                self.data = pd.read_csv(".csv")
                self.audios = ""
            else:
                self.data = pd.read_csv(".csv")
                self.audios = ""
        elif self.split == "val":
            if(gender == 'f'):
                self.data = pd.read_csv(".csv")
                self.audios = ""
            else:
                self.data = pd.read_csv(".csv")
                self.audios = ""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: leggere la label associata al file audio nel CSV
        label = self.data.loc[idx, "emotion"]
        label = int(label)

        # TODO: leggere il percorso del file audio dal CSV
        img_name = self.data.loc[idx, "audio_path"]

        # TODO: creare lo spettrogramma e salvarlo nell'oggetto img
        img = None

        if self.transform is not None:
            img = self.transform(img)

        return {'image': img, 'label': label}


if __name__ == "__main__":
    split = "train"
    demos_train = Demos(split=split)
    print("Demos {} set loaded".format(split))
    print("{} samples".format(len(demos_train)))

    for i in range(3):
        print(demos_train[i]["label"])
        demos_train[i]["image"].show()

