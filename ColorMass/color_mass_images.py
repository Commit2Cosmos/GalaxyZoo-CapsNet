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
                                                                     'SDSS_petromag_z',
                                                                     't01_smooth_or_features_a01_smooth_weighted_fraction',
                                                                     't01_smooth_or_features_a02_features_or_disk_weighted_fraction'
                                                                    ])


# cm_csv = pd.read_parquet(f'../Data/gz2hart_mpajhu.parquet')
# print(cm_csv.columns.values)

match_test_csv = pd.read_csv(f'../Data/match_test_kaggle.csv', usecols=['GalaxyID', 'dr7objid', 'Usage'])
# match_test_csv = match_test_csv[match_test_csv['Usage'] != 'private']
# match_test_csv = match_test_csv[match_test_csv['Usage'] != 'public']
match_test_csv.drop(columns=['Usage'], inplace=True)
# print(match_test_csv)

merged = pd.merge(left=match_test_csv, right=cm_csv, left_on='dr7objid', right_on='DR7_GZ_OBJID')
merged.drop(columns=['dr7objid', 'DR7_GZ_OBJID'], inplace=True)


merged = merged[(merged['MPAJHU_Z'] > 0.02) & (merged['MPAJHU_Z'] < 0.05)]
merged.drop(columns=['MPAJHU_Z'], inplace=True)



merged = merged[merged['SDSS_petromag_z'] < 19.5]
merged.drop(columns=['SDSS_petromag_z'], inplace=True)

merged['SDSS_petromag_abs_u_kcorr_dustcorr'] = merged['SDSS_petromag_abs_u_kcorr_dustcorr'] - merged['SDSS_petromag_abs_r_kcorr_dustcorr']
merged.drop(columns=['SDSS_petromag_abs_r_kcorr_dustcorr'], inplace=True)

#* get only definitely labeled
merged = merged[(merged['t01_smooth_or_features_a01_smooth_weighted_fraction'] > 0.8) | (merged['t01_smooth_or_features_a02_features_or_disk_weighted_fraction'] > 0.8)]
merged = merged[merged['MPAJHU_MEDIAN_MASS'] > 2.0]
# merged = merged[merged['SDSS_petromag_abs_u_kcorr_dustcorr'] > -1.0]


#* match labels with images
# import os
# merged['GalaxyID'] = merged.iloc[:,0].map(lambda x:f'{x}.jpg')
# filenames = os.listdir('../Data/images_train_kaggle')
# filenames = pd.DataFrame(filenames, columns=['filenames'])
# merged = pd.merge(left=merged, right=filenames, left_on='GalaxyID', right_on='filenames')
# merged.drop('filenames', axis=1, inplace=True)


#! GZ labels based selection
# Early type smooth condition
# merged = merged[merged['t01_smooth_or_features_a01_smooth_weighted_fraction'] > 0.8]
# Late type spiral condition
# merged = merged[merged['t01_smooth_or_features_a02_features_or_disk_weighted_fraction'] > 0.8]

merged.drop(columns=['t01_smooth_or_features_a01_smooth_weighted_fraction'], inplace=True)
merged.drop(columns=['t01_smooth_or_features_a02_features_or_disk_weighted_fraction'], inplace=True)

print(merged)

#* Save original color & mass
original_color_mass = np.array(merged.iloc[:, 1:])
np.save(f'./ColorMass/color_mass_original', original_color_mass)


#* Save csv for Dataloader to input into Predictor
# merged.to_csv(f'./ColorMass/color_mass_original.csv', index=False)