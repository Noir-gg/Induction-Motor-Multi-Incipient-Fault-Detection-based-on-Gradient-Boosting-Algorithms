import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("../datasets/dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, 1:dataset.shape[1]], \
                                                    dataset["label"], test_size=0.20, random_state=42)


X_train = np.array(X_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)

print(X_train)

feature1_index = 1  # Replace 0 with the index of the first feature
feature2_index = 2  # Replace 1 with the index of the second feature

plt.figure(figsize=(10, 5))
colors = ['blue', 'orange', 'green']
labels = ['Class 0', 'Class 1', 'Class 2']

plt.plot(X_train[0])
plt.xlabel('X_train[0]')
plt.ylabel('X_train[1]')
plt.title('Training Data Scatter Plot')
plt.legend()
plt.show()