import pandas as pd
import numpy as np


# match_test_csv = pd.read_csv(f'../Data/match_test_kaggle.csv', usecols=['GalaxyID', 'dr7objid'])
# votes_train_csv = pd.read_csv(f'../Data/labels_train_kaggle.csv', usecols=['GalaxyID', 'Class1.1', 'Class1.2'])
# votes_train_csv = pd.merge(left=match_test_csv, right=votes_train_csv, on='GalaxyID')
# votes_train_csv.drop(columns=['GalaxyID'], inplace=True)

# print(votes_train_csv)


# ignore
# cm_csv = pd.read_csv('../Data/gz2_kaggle_hart_mpajhu_match.csv', usecols=['GalaxyID_Kaggle', 'Z', 'PETROMAG_U', 'PETROMAG_R', 'PETROMAG_Z', 'LGM_TOT_P50'])


# cols: MPAJHU_Z, DR7_GZ_PETROMAG_Z / DR7_GZ_EXTINCTION_Z / DR7_GZ_PETROMAG_MZ / SDSS_petromag_z ,
# SDSS_petromag_abs_u_kcorr_dustcorr , SDSS_petromag_abs_r_kcorr_dustcorr, MPAJHU_MEDIAN_MASS
cm_csv = pd.read_parquet(f'../Data/gz2hart_mpajhu.parquet', columns=[
                                                                     'DR7_GZ_OBJID',
                                                                     'MPAJHU_Z',
                                                                     'SDSS_petromag_abs_u_kcorr_dustcorr',
                                                                     'SDSS_petromag_abs_r_kcorr_dustcorr',
                                                                     'MPAJHU_MEDIAN_MASS',
                                                                     't01_smooth_or_features_a01_smooth_weighted_fraction',
                                                                     't01_smooth_or_features_a02_features_or_disk_weighted_fraction'
                                                                    ])



# cm_csv = pd.read_parquet(f'../Data/gz2hart_mpajhu.parquet')
# print(cm_csv.columns.values)

match_test_csv = pd.read_csv(f'../Data/match_test_kaggle.csv', usecols=['GalaxyID', 'dr7objid'])

merged = pd.merge(left=match_test_csv, right=cm_csv, left_on='dr7objid', right_on='DR7_GZ_OBJID')
merged.drop(columns=['dr7objid', 'DR7_GZ_OBJID'], inplace=True)


merged = merged[(merged['MPAJHU_Z'] > 0.02) & (merged['MPAJHU_Z'] < 0.05)]
# # print(merged['MPAJHU_Z'])
merged.drop(columns=['MPAJHU_Z'], inplace=True)



#* use < -19.8 SDSS_petromag_abs_r_kcorr_dustcorr

merged = merged[merged['SDSS_petromag_abs_r_kcorr_dustcorr'] < -19.5]
merged['SDSS_petromag_abs_u_kcorr_dustcorr'] = merged['SDSS_petromag_abs_u_kcorr_dustcorr'] - merged['SDSS_petromag_abs_r_kcorr_dustcorr']
merged.drop(columns=['SDSS_petromag_abs_r_kcorr_dustcorr'], inplace=True)

#* get only definitely labeled
# merged = merged[(merged['t01_smooth_or_features_a01_smooth_weighted_fraction'] > 0.8) | (merged['t01_smooth_or_features_a02_features_or_disk_weighted_fraction'] > 0.8)]
merged = merged[merged['MPAJHU_MEDIAN_MASS'] > 2.0]
merged = merged[(merged['SDSS_petromag_abs_u_kcorr_dustcorr'] > 0.8) & (merged['SDSS_petromag_abs_u_kcorr_dustcorr'] < 3.25)]


#* match labels with images
import os
merged['GalaxyID'] = merged.iloc[:,0].map(lambda x:f'{x}.jpg')
filenames = os.listdir('../Data/images_train_kaggle')
filenames = pd.DataFrame(filenames, columns=['filenames'])
merged = pd.merge(left=merged, right=filenames, left_on='GalaxyID', right_on='filenames')
merged.drop('filenames', axis=1, inplace=True)


#! GZ labels based selection
#* TOTAL: 23582
#* Early type smooth condition: 7159
# merged = merged[merged['t01_smooth_or_features_a01_smooth_weighted_fraction'] > 0.5]
#* Late type spiral condition: 5189
# merged = merged[merged['t01_smooth_or_features_a02_features_or_disk_weighted_fraction'] > 0.5]

merged.drop(columns=['t01_smooth_or_features_a01_smooth_weighted_fraction'], inplace=True)
merged.drop(columns=['t01_smooth_or_features_a02_features_or_disk_weighted_fraction'], inplace=True)



#! CapsNet predictions based selection

#* Get predictions from CapsNet
# predictions = np.load(f'./ColorMass/preds/preds_simard_grey.npy')
# print(predictions)

#* add predictions as separate column to df
# merged['CapsNetPreds'] = predictions.tolist()

#* Early type smooth condition: kaggle:4330 / 5852 | simard:11937 / 12322
# merged = merged[merged['CapsNetPreds'] <= 0.2]
#* Late type spiral condition: kaggle:16984 / 15260 | simard:10732 / 10472
# merged = merged[merged['CapsNetPreds'] >= 0.8]

# merged.drop(columns=['CapsNetPreds'], inplace=True)



#! DeepCaps predictions based selection

#* Get predictions from DeepCaps
predictions = np.load(f'./ColorMass/preds/preds_kaggle_deepcaps.npy')

#* add predictions as separate column to df
merged['DeepCapsSmoothPreds'] = predictions[:,1].tolist()
merged['DeepCapsFeaturedPreds'] = predictions[:,0].tolist()

#* Early type smooth condition: kaggle:7526 | simard:13061           NO GALAXIES WITH BOTH FRACTIONS > 0.8
# merged = merged[merged['DeepCapsSmoothPreds'] >= 0.97]
#* Late type spiral condition: kaggle:14947 | simard:11091
merged = merged[merged['DeepCapsFeaturedPreds'] >= 0.95]

merged.drop(columns=['DeepCapsFeaturedPreds'], inplace=True)
merged.drop(columns=['DeepCapsSmoothPreds'], inplace=True)


print(merged)
print(merged.describe())


#! Save original color & mass

#* Save csv for Dataloader to input into Predictor
# merged.to_csv(f"./ColorMass/color_mass.csv", index=False)

#* Save npy for Colour_Mass_Plot
color_mass = np.array(merged.iloc[:, 1:])
np.save(f'./ColorMass/color_mass_labels', color_mass)