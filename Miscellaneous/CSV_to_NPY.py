import pandas as pd
import numpy as np

df = pd.read_csv('./PreparedData/Kaggle/paths_votes.csv')

X = np.array(df.iloc[:, 1:])

print(X)
print(X.shape)

# np.save('./PreparedData/Kaggle/Grey/votes', X)
np.save('./PreparedData/Kaggle/RGB/all_votes', X)