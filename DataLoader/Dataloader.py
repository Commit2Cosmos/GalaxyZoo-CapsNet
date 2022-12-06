from email.mime import image
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms 
import PIL
import numpy as np



"""
We will read the csv in __init__ but leave the
reading of images to __getitem__
"""

class SDSSData(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """ 
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform 



    #len(dataset) returns the size of the dataset
    #The __len__ function returns the number of samples in our dataset
    def __len__(self):
        return len(self.annotations) #number of images/Entries in csv file


    #This will return a given image and a corrosponding index for the image
    #__getitem__ to support the indexing such that dataset[i] can be used to get ith sample.
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image1 = io.imread(img_path)
        labels = torch.tensor(self.annotations.iloc[index, 1:])

        if self.transform:
            image = self.transform(image1)
        return (image)



#in main code:
#'../gz_decals_dr5'
# transformed_dataset = SDSSData(csv_file='../SDSS/Data/61000Sample.csv', 
# root_dir='../SDSS/images', 
# transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.Grayscale(num_output_channels=1), transforms.ToPILImage()]))

transformed_dataset = SDSSData(csv_file='/mmfs1/home/users/belov/Data/paths_votes.csv', 
root_dir='/mmfs1/home/users/belov/Data/images_gz2',
transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.Grayscale(num_output_channels=2), transforms.ToPILImage()]))

list=[]
for i in range(len(transformed_dataset)):
    images = transformed_dataset[i]
    # npimages = images.numpy()
    npimages = np.array(images)
    list.append(npimages)
    print(i)

# list = np.array(list)
# print(list.size)
np.save('/mmfs1/home/users/belov/ReadyFile/images', list)