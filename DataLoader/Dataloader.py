from email.mime import image
import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import cv2



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


# Simard or Kaggle
DATASET = 'Simard'

transformed_dataset = SDSSData(
csv_file=f'./PreparedData/{DATASET}/paths_votes_6.csv',
root_dir='../Data/images_train_kaggle',     #! DON'T CHANGE
# transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.Grayscale(num_output_channels=1)]))
# transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.Grayscale(num_output_channels=1), transforms.ToPILImage()]))
# transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), GaussianBlurAugmentation()]))
transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72))]))


toPIL = transforms.ToPILImage()


#*#######  VIEW 4 IMAGES FROM DATASET  ############
# image_index = 20
# for i in range(image_index, image_index+5):
#     sample = transformed_dataset[i]
#     sample = torch.squeeze(sample)
#     # print(sample.shape)
#     sample = toPIL(sample)

#     ax = plt.subplot(1, 4, i+1)
#     plt.tight_layout()
#     ax.set_title('Sample {}'.format(i+1))
#     ax.axis('off')
#     plt.imshow(sample)
# plt.show()

#* PCA on 1 image
# NUM_COLORES = 3
# def apply_pca(img):
    #* Standardize
    # img = img.reshape(-1, NUM_COLORES)
    # img = np.array(img)

    # matrix = np.zeros((img[:,0].size, NUM_COLORES))

    # for i in range(NUM_COLORES):
    #     dim_arr = img[:,i]
    #     arrayStd = (dim_arr - dim_arr.mean())/dim_arr.std()
    #     matrix[:,i] = arrayStd

    #* Compute eigen-values/vectors

    # cov = np.cov(matrix.transpose())

    # EigVal, EigVec = np.linalg.eig(cov)
    # print(EigVal)

    #* Sort vectors by values
    # order = EigVal.argsort()[::-1]
    # EigVal = EigVal[order]
    # EigVec = EigVec[:,order]

    #* Projecting data on Eigen vector directions resulting to Principal Components (cross product)
    # PC = np.matmul(matrix, EigVec)

    #* Dependency check (Generate Pairplot for original data and transformed PCs)
    #* Original
    # COLORES = ['R', 'G', 'B']
    # a = sns.pairplot(pd.DataFrame(matrix, columns = COLORES))
    # a.fig.suptitle("Pair plot of Band images")

    #* PC
    # PCs = ['PC 1', 'PC 2', 'PC 3']
    # a = sns.pairplot(pd.DataFrame(PC, columns = PCs))
    # a.fig.suptitle("Pair plot of PCs")

    # plt.show()

    # plt.figure(figsize=(8,6))
    # plt.bar([1,2,3], EigVal/sum(EigVal)*100, align='center', width=0.4, tick_label = PCs)
    # plt.ylabel('Variance (%)')
    # plt.title('Information retention')
    # plt.show()

    #* Convert back to images
    # PC = PC.reshape(-1, 216, 216)
    # img = img.reshape(-1, 216, 216)
    
    #* Visualize compression
#     fig, axes = plt.subplots(1, 3, figsize=(10,7))
#     fig.subplots_adjust(wspace=0.1, hspace=0.15)
#     fig.suptitle('Intensities of Principal Components ', fontsize=12)

#     axes = axes.ravel()
#     for i in range(NUM_COLORES):
#         PC_img = PC[i,:,:]
#         axes[i].imshow(PC_img, cmap='gray')
#         axes[i].set_title('PC ' + str(i+1), fontsize=12)
#         axes[i].axis('off')
#     plt.show()

# apply_pca(transformed_dataset[0])




#*#######  VIEW ORIGINAL VS TRANSFORMED  #############
# samples = 4
# starting_index = 28

# fig, axs = plt.subplots(2, samples, figsize=(10,7))
# fig.subplots_adjust(wspace=0.1, hspace=0.0)
# axs = axs.ravel()

# for i in range(starting_index, starting_index + samples):
#     original = transformed_dataset[i]
#     # transformed = transforms.Resize((72,72))(original)
#     transformed = apply_pca(original)
    
#     ii = (i-starting_index)

#     axs[ii].imshow(toPIL(original))
#     axs[ii].axis('off')

#     axs[ii+samples].imshow(toPIL(transformed))
#     axs[ii+samples].axis('off')

# plt.show()



list = []
for i in range(len(transformed_dataset)):
    images = transformed_dataset[i]
    npimages = np.array(images)
    list.append(npimages)
    print(i)


# np.save(f'./PreparedData/{DATASET}/RGB/images_6', list)