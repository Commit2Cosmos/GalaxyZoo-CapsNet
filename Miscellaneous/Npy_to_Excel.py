import pandas as pd
import numpy as np

# array = np.load(r"C:\Users\USER\Documents\____MastersProject\GreenValleyData\KaggleRGBREDUCEPredict.npy", allow_pickle=True)
array = np.load(r"C:\Users\Anton (Main)\Desktop\Uni\Phys4xx\!Masters451\Network\AlexNetwork\losses\RGBtrain_losses.npy", allow_pickle=True)

print(array.shape)
#print(array)


## convert your array into a dataframe
#array will be a 3-D shape of N,2,37 in dimension, reshape to (2N,37)
# df = pd.DataFrame (array.reshape(11408, 37))
df = pd.DataFrame(array)

## save to xlsx file

filepath = (r"C:\Users\Anton (Main)\Desktop\Uni\Phys4xx\!Masters451\Network\AlexNetwork\lossesFile\train_losses.xlsx")

df.to_excel(filepath, index=False)