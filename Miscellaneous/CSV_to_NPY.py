import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Anton (Main)\Desktop\!extracted\imageAnnotations.csv")

X =np.array(df.iloc[:, 1:])

print(X)
print(X.shape)

np.save(r"C:\Users\Anton (Main)\Desktop\Uni\Phys4xx\!Masters451\Network\AlexNetwork\savedCSV.npy", X)

#print(X[60891])