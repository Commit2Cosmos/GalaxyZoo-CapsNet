import pandas as pd
import numpy as np

array = np.load('./Results/test_losses.npy', allow_pickle=True)

print(array.shape)
print(array)


# convert your array into a dataframe
# array will be a 3-D shape of N,2,37 in dimension, reshape to (2N,37)
# df = pd.DataFrame (array.reshape(11408, 37))
# df = pd.DataFrame (array.reshape(11408, 2))

# save to xlsx file

# filepath = ('../plots/test_losses.xlsx')

# df.to_excel(filepath, index=False)