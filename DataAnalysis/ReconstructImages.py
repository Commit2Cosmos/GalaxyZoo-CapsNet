import numpy as np
import matplotlib.pyplot as plt

# 10, 50, 100 or 200
EPOCH = 100

# Y=np.load(f'\\Users\\Anton\\Desktop\\!Uni\\Phys4xx\\!Masters451\\Network\\HECResults\\Kaggle\\2Params\\Epochs\\Truth\\epoch_{EPOCH}.npy')
Z=np.load(f'../HECResults/Kaggle/2Params/RGB/Epochs(4)/Recon/epoch_{EPOCH}.npy', allow_pickle=True)
print(Z.shape)

# Original images
# for i, el in enumerate(Y):
#     #moving axis to use plt: i.e [4,100,100] to [100,100,4]
#     array2 = np.moveaxis(Y[i], 0, -1)
#     plt.subplot(4, 5, i + 1)
#     plt.imshow(array2, cmap ='gray', interpolation='nearest', ) 
#     plt.axis('off')

# plt.suptitle('Original Images')  
# plt.show()

#* Reconstructed images
# for i, el in enumerate(Z):
#     #moving axis to use plt: i.e [4,100,100] to [100,100,4]
#     array3 = np.moveaxis(Z[i], 0, -1)
#     plt.subplot(4, 5, i + 1)
#     plt.imshow(array3, cmap ='gray')
#     plt.axis('off')
#     #plt.subplots_adjust(wspace=0, hspace=0) 

# plt.suptitle(f'Reconstruction after {EPOCH} epochs')    
# plt.show()