import torch.utils.data as data
import pandas as pd
from PIL import Image


class Emovo(data.Dataset):

    # TODO: implementare come in demos.py

    def __init__(self, split='train', transform=None):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass


if __name__ == "__main__":
    pass