from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch

class cats_dogs_dataset(Dataset):
    
    '''
    construct custom dataset for cat-dog binary classification
    '''

    def __init__(self, image_dir, annotations, transform=None):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(annotations)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        '''
        index through the dataset and build it
        '''
        # first, we get image id from annotations
        img_id = self.annotations.iloc[index, 0]
        
        # load the image
        img = Image.open(os.path.join(self.image_dir, img_id)).convert('RGB')
        
        # get the correct semantic
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, y_label