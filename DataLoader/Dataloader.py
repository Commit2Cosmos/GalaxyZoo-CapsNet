from email.mime import image
import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import torch

import numpy as np
# from skimage.filters import threshold_otsu, gaussian

# Simard or Kaggle
DATASET = 'Simard'


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
        image = io.imread(img_path)
        # labels = torch.tensor(self.annotations.iloc[index, 1:])

        if self.transform:
            image = self.transform(image)
        return image

        # if self.transform:
        #     image = self.transform(image)
        #     Grayimg1 = img_as_float(image)
        #     Grayimg = gaussian(Grayimg1, sigma=2.25)
        #     #Grayimg = gaussian(Grayimg1, sigma=3)
        #     transformTensor=transforms.Compose([transforms.ToTensor()])
        #     thresh = threshold_otsu(Grayimg)
        #     binary = Grayimg > thresh
        #     GrayimgTensor = transformTensor(Grayimg1)
        #     binaryTensor = transformTensor(binary)
        #     Segmented = torch.mul(GrayimgTensor, binaryTensor)
        # return Segmented

class GaussianBlurAugmentation:
    """Blurs image with randomly chosen Gaussian blur
    """
    def __call__(self, sample):
        noise_factor = 0.3
        torch.tensor(sample)
        noisy = sample + torch.rand_like(sample) * noise_factor
        noisy = torch.clamp(noisy, 0., 1.)
        noisy = np.array(noisy)
        return noisy


transformed_dataset = SDSSData(
csv_file=f'./PreparedData/{DATASET}/paths_votes_test.csv', 
#* DON'T CHANGE
root_dir='../Data/images_train_kaggle',
# transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.Grayscale(num_output_channels=1)]))
# transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.Grayscale(num_output_channels=1), transforms.ToPILImage()]))
# transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), GaussianBlurAugmentation()]))
transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72))]))


# for i in range(4):
#     sample = transformed_dataset[i]
#     sample = torch.squeeze(sample)
#     # print(sample.shape)
    
#     ax = plt.subplot(1, 4, i+1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample)
# plt.show()


list = []
for i in range(len(transformed_dataset)):
    images = transformed_dataset[i]
    npimages = np.array(images)
    list.append(npimages)
    print(i)


np.save(f'./PreparedData/{DATASET}/RGB/images_test', list)