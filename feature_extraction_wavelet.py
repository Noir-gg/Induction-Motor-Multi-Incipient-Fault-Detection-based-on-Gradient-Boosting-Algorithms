import pandas as pd
import numpy as np
import pywt
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("../datasets/dataset_resampled.csv")
degree_of_transforms = [20, 38, 17]
wavelet_types = ["sym", "db", "coif"]

def detailed_approximate_coefficients(current, wavelet_type, degree_of_transform, index):

    wavelet = wavelet_type + str(degree_of_transform)
    _, cD_data = pywt.dwt(current, wavelet=wavelet)

    return np.array(cD_data)

class_0_data = dataset[dataset['label'] == 0] 
class_1_data = dataset[dataset['label'] == 1]  
class_2_data = dataset[dataset['label'] == 2] 

class_0_array = class_0_data.drop('label', axis=1).to_numpy()
class_1_array = class_1_data.drop('label', axis=1).to_numpy()
class_2_array = class_2_data.drop('label', axis=1).to_numpy()

class_0_array = class_0_array.reshape(-1, 1)
class_1_array = class_0_array.reshape(-1, 1)
class_2_array = class_0_array.reshape(-1, 1)

currents = [class_0_array, class_1_array, class_2_array]

df = pd.DataFrame()
for index, current in enumerate(currents):
    temp_df = pd.DataFrame()
    label = f'{index}'
    for wavelet_type in wavelet_types:
        degree_of_transform = wavelet_types.index(wavelet_type)
        degree_of_transform = degree_of_transforms[degree_of_transform]
        cD_data = detailed_approximate_coefficients(current, wavelet_type, degree_of_transform, index)
        wavelets = [wavelet_type+str(i) for i in range(1,cD_data.shape[1]+1)]
        cD = pd.DataFrame(cD_data, columns = wavelets)
        if wavelet_type == "sym":
            cD.insert(0, 'label', label)
        temp_df = pd.concat([temp_df, cD], axis=1)
    df = pd.concat([df,temp_df], axis=0)

X, y = df.drop(columns =['label']), df['label']

print(f"Initial number of features: {X.shape[1]}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features normalized.")

mi_scores = mutual_info_classif(X, y)
print(mi_scores)
print("MI Calculation complete; thresholding...")
top_percent_idx = np.argsort(mi_scores)[-int(len(mi_scores) * 0.10):]
high_mi_features = X.columns[top_percent_idx]
print(high_mi_features)

result= pd.DataFrame(df['label'])
for wavelet in high_mi_features:
    result[wavelet] = df[wavelet]
result.to_csv("../datasets/wavelet_undersamplified.csv", index=False)