"""Class for custom Youtube thumbnails image dataset
"""
import os
import pandas
import glob
import pandas as pd
import time
from PIL import Image
from torch.utils.data import Dataset

class YoutubeDataset(Dataset):
    """Youtube image dataset. This contains no labels.
    """
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        img_list = [x[-15:] for x in glob.glob(self.root_dir + '*.jpg')]
        all_annotations = pd.read_csv(csv_file)
        self.annotations = all_annotations[all_annotations['img_name'].isin(img_list)].reset_index(drop=True)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.annotations['img_name'][idx]
        img_path = self.root_dir + img_name
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        # Load label
        label = self.annotations['label'][idx]
        
        return {'image': image, 'label': label}