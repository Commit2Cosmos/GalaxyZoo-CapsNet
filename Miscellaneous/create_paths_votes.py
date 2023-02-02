import pandas as pd
import os
import math
import numpy as np


#* load votes
merged = pd.read_csv('./InitData/Kaggle/labels_train.csv', usecols=[0,1,2,3])
# merged = pd.read_csv('./InitData/Kaggle/labels_train.csv')


#* calculate agreement parameter
def calculate_agreement(row):
    
    res = 0

    for i in range(1, 4):
        item = row[i]
        if item != 0:
            res += item * math.log(item)
    
    res = abs(res)

    agreement = 1 - res/math.log(3)

    return agreement

#* Remove galaxies based on agreement (for binary)
merged['Agreement'] = merged.apply(lambda row: calculate_agreement(row), axis=1)
merged = merged[merged['Agreement'] >= 0.8]


#* make column with max value == 1, otherwise 0 (for binary)
m = np.zeros_like(merged.iloc[:, 1:-1].values)
m[np.arange(len(m)), merged.iloc[:, 1:-1].values.argmax(1)] = 1
merged.iloc[:, 1:-1] = m
sums = merged.iloc[:, 1:-1].sum(axis=0)
print(f"Smooth count: {int(sums[0])}")
print(f"Featured or disk count: {int(sums[1])}")
print(f"Star/artifact count: {int(sums[2])}")

#* remove artifacts column if empty
if int(sums[2]) == 0:
    merged.drop('Class1.3', axis=1, inplace=True)
    merged.drop('Class1.2', axis=1, inplace=True)
    merged.drop('Agreement', axis=1, inplace=True)
else:
    raise Exception('There are artifacts in the dataset!')


#* MATCH SDSS & IMAGE IDS FOR SIMARD
# ids = pd.read_csv('./InitData/Kaggle/match_ids.csv', usecols=[0,1])
# merged = pd.merge(left=merged, right=ids, on='GalaxyID', how='inner')
# print(ids)


#* match labels to images 
merged['GalaxyID'] = merged.iloc[:102,0].map(lambda x:f'{x}.jpg')
# filenames = os.listdir('./InitData/Kaggle/images_train')
filenames = os.listdir('../Data/images_train_kaggle')
filenames = pd.DataFrame(filenames, columns=['filenames'])
merged = pd.merge(left=merged, right=filenames, left_on='GalaxyID', right_on='filenames')
merged.drop('filenames', axis=1, inplace=True)
print(merged)


#* save csv
merged.to_csv('./PreparedData/Kaggle/paths_votes_2.csv', encoding='utf-8', index=False)