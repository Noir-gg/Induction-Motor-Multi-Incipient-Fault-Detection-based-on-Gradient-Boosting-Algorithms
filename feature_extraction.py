import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.stats import skew
from imblearn.under_sampling import RandomUnderSampler

def apply_smote(dataset):
    smote_columns = ['Phase A', 'Phase B', 'Phase C']
    X = dataset[smote_columns].values
    y = dataset['label'].values                                                       
    X_resampled, y_resampled = smote.fit_resample(X, y)
    resampled_df = pd.DataFrame(X_resampled, columns=smote_columns)
    resampled_df['label'] = y_resampled
    cols = ['label'] + smote_columns
    resampled_df = resampled_df[cols]
    resampled_df = resampled_df.sort_values(by='label')
    return resampled_df

smote = SMOTE()

dataset1 = pd.read_excel("../datasets/bearing_fault_dataset.xlsx")
dataset2 = pd.read_excel("../datasets/normal_dataset.xlsx")
dataset3 = pd.read_excel("../datasets/stator_winding_fault_dataset.xlsx")
dataset = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)

dataset_resampled = apply_smote(dataset)
dataset_resampled.to_csv(os.path.join("../datasets/dataset_resampled.csv"), index=False)

skew_values = []
covariance_values = []
mean_values = []
variance_values = []
std_deviation_values = []
fft_A = []
fft_B = []
fft_C = []
length = dataset_resampled.shape[0]

for index, row in dataset_resampled.drop("label", axis=1).iterrows():

    skew_values.append(row.skew())

    covariance_values.append(row.cov(dataset_resampled.drop("label", axis=1).iloc[index]))

    mean_values.append(pd.Series(row).mean())
    variance_values.append(pd.Series(row).var())
    std_deviation_values.append(pd.Series(row).std())
    print(length)
    length -= 1

dataset_resampled['skew'] = skew_values
dataset_resampled['covariance'] = covariance_values
dataset_resampled['mean'] = mean_values
dataset_resampled['variance'] = variance_values
dataset_resampled['std_deviation'] = std_deviation_values

dataset_resampled.to_csv(os.path.join("../datasets/dataset.csv"), index=False)