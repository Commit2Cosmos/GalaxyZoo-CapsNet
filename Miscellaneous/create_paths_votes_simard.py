import pandas as pd
import os
import numpy as np
import torch


#* MATCH SDSS & IMAGE IDS FOR SIMARD
# ids = pd.read_csv(f'\\Users\\Anton\\Desktop\\!Uni\\Phys4xx\\!Masters451\\Network\\Data\\match_test_kaggle.csv', usecols=[0,1])


#* match ids to previous dataset by usage (either this or ids_required)
ids = pd.read_csv(f'\\Users\\Anton\\Desktop\\!Uni\\Phys4xx\\!Masters451\\Network\\Data\\match_test_kaggle.csv')
ids = ids[ids['Usage'] == 'training']
ids.drop('Usage', axis=1, inplace=True)


#* match kaggle ids to previous dataset
# ids_required = pd.read_csv('./InitData/Kaggle/labels_train.csv', usecols=[0])
# ids = pd.merge(left=ids, right=ids_required, on='GalaxyID', how='inner')


#* load votes
votes = pd.read_csv(f'\\Users\\Anton\\Desktop\\!Uni\\Phys4xx\\!Masters451\\Network\\Data\\simard_nfree.csv', 
usecols=['objID', 'rg2d', 'Rchl_r', 'e', 'phi', 'S2r', 'ng'])


#* match kaggle & SDSS ids
votes = pd.merge(left=votes, right=ids, left_on='objID', right_on='dr7objid', how='inner')
votes.drop(['objID', 'dr7objid'], axis=1, inplace=True)
cols = votes.columns.tolist()
cols = cols[-1:] + cols[:-1]
votes = votes[cols]


#* check rows for NaN values & delete all rows containing them
# print(votes[votes.isnull().any(axis=1)])
votes = votes.dropna(axis = 0)


#* convert Rhlr to arsec
# votes['Rhlr'] = votes['Rhlr'] * votes['Scale']
# votes.drop('Scale', axis=1, inplace=True)


#* convert phi to rads
# votes['phi'] = votes['phi'] * math.pi/180


#* check min & max values in every column
# print(pd.DataFrame(votes.values.min(0)[None, :], columns=votes.columns))
# print(pd.DataFrame(votes.values.max(0)[None, :], columns=votes.columns))


#! MULTI PARAMETER ONLY ! #######################################

#* rescaling so all label values are in set [0,1]
# minim = votes.iloc[:,1:].min()
# maxim = votes.iloc[:,1:].max()
# for i in range(1, len(votes.columns)):
#     votes.iloc[:,i] = (votes.iloc[:,i] - minim[i-1]) / (maxim[i-1] - minim[i-1])


#* 


#* PCA
# labels = torch.tensor(votes.iloc[:5,1:3].values, dtype=torch.float32)
# print(labels)
# U, S, V = torch.pca_lowrank(labels, niter=3)
# print(V)
# print(votes)

#!#############################################################



#! 2 PARAMETER ONLY ! #######################################

# #* find relative error & filter accordingly


# #* 0: smooth (n > 3); 1: featured (n < 1.5)
# votes = votes.loc[(votes['ng'] > 3.0) | (votes['ng'] < 1.5)]
# votes.loc[votes['ng'] < 1.5, 'ng'] = 1
# votes.loc[votes['ng'] > 3, 'ng'] = 0


# #* filter out unphysical half-light radius value
# votes = votes.loc[(votes['Rchl_r'] > 0.5) & (votes['Rchl_r'] < 50.0)]


# #* remove all but first and last column
# votes = votes.filter(['GalaxyID', 'ng'])

#!#############################################################


#* match labels to images
# votes['GalaxyID'] = votes.iloc[:,0].map(lambda x:f'{x}.jpg')
# filenames = os.listdir('../Data/images_train_kaggle')
# filenames = pd.DataFrame(filenames, columns=['filenames'])
# votes = pd.merge(left=votes, right=filenames, left_on='GalaxyID', right_on='filenames')
# votes.drop('filenames', axis=1, inplace=True)


#* print total counts of featured and smooth samples
# total = len(votes.index)
# feaured = int(votes['ng'].sum(axis=0))
# print(f"Featured count: {feaured}")
# print(f"Smooth count: {total - feaured}")


#* save csv
# votes.to_csv('./PreparedData/Simard/paths_votes_6.csv', encoding='utf-8', index=False)