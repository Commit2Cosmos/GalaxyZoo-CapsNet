# ImageFilter for using filter() function
from PIL import Image, ImageFilter
from skimage import io
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage import data, img_as_float
from torchvision import transforms 
import numpy as np
import matplotlib.pyplot as plt

image = Image.open('./InitData/Kaggle/images_train/100008.jpg')

# transform=transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.ToTensor()])
transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.Grayscale(num_output_channels=1), transforms.ToPILImage()])
transformRGB=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((216,216)), transforms.Resize((72,72)), transforms.ToPILImage()])
transformTensor=transforms.Compose([transforms.ToTensor()])


org_img = transformTensor(image)
array = np.reshape(org_img, (424, 424, -1))
# array = array.transpose(0, 1)
print(array.shape)
#Display original image
#plt.imshow(array, cmap=plt.cm.gray)
plt.tight_layout()
plt.imshow(array)
plt.axis('off')
plt.tight_layout()
plt.show()

transformed = transform(image)

# Grayimg = img_as_float(transformed)
# gau_img = gaussian(Grayimg, sigma=3)

#Display gaussian blurred image
# array = np.reshape(gau_img, (72, 72))
# plt.imshow(transformed, cmap=plt.cm.gray)
plt.imshow(transformed)
#plt.imshow(array)
plt.axis('off')
plt.show()