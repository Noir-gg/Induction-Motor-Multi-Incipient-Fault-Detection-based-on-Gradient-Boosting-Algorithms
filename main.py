import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import fftpack
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, make_scorer
import time
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, PassiveAggressiveClassifier, Perceptron, SGDClassifier
from xgboost import XGBClassifier
from scipy.spatial.distance import pdist, squareform
from itertools import combinations_with_replacement
from sklearn.semi_supervised import SelfTrainingClassifier
from datetime import datetime
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

class_names = ['normal', 'bearing fault', 'stator winding fault']  
num_classes = len(class_names)

models = [ 

    ("GBM", GradientBoostingClassifier()),

    ("CatBoost", CatBoostClassifier(iterations=250, learning_rate=0.15, depth=15, loss_function='MultiClass')),

]

accuracies = []
f1_scores = []

cv_accuracies = []
cv_f1_scores = []
results_dict = {
    "Model": [],
    "Accuracy": [],
    "F1": [],
    "Precision": [],
    "Recall": [],

    "Accuracy_CV": [],
    "runtime": [],

}

results_root_dir = "../results/"
results_file = os.path.join(results_root_dir, "results.xlsx")

dataset = pd.read_csv("../datasets/dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, 1:dataset.shape[1]],\
                                                    dataset["label"], test_size=0.20, random_state=42)

for model_name, model in models:
    start_time = time.time()
    print("Running", model_name)
    model.fit(X_train, y_train)

    cv_results = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_accuracy = cv_results.mean()

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)                                  
    print(accuracy)
    precision = precision_score(y_test, y_pred,average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    results_dir = results_root_dir + model_name + "/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if model_name != "PassiveAggressive" and model_name != "Perceptron" and model_name != "NearestCentroid":

        y_pred_proba = model.predict_proba(X_test)
        num_classes = len(np.unique(y_train))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve for Multiclass Classification')
        plt.legend(loc="lower right")
        plt.legend()
        plt.savefig(fname=os.path.join(results_dir, model_name + "_ROC.png"))
        plt.close()

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)  
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(fname=os.path.join(results_dir,model_name + "_Confusion Matrix.png"))
    plt.close()

    results_dict["Model"].append(model_name)
    results_dict["Accuracy"].append(accuracy)
    results_dict["F1"].append(f1.mean())
    results_dict["Precision"].append(precision.mean())
    results_dict["Recall"].append(recall.mean())
    results_dict["Accuracy_CV"].append(mean_accuracy)

    model_time = time.time()
    model_time -= start_time
    model_time /= 60
    results_dict["runtime"].append(model_time)
    print(f"Model {model_name} ended in {model_time:.2f} minutes")

results_df = pd.DataFrame(results_dict)
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"{current_datetime}_results.csv"
results_df.to_csv(os.path.join(results_root_dir,"metrics", file_name), index=False)

end_time = time.time()
execution_time = end_time - start_time
execution_time /= 60
print("Modelling Completed")
print("Total execution time: {:.2f} minutes".format(execution_time))
