import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from datetime import datetime

knn_neighbours =  [3]

XGB_lr = [0.5]
xgb_reg_lambda = [3]
xgb_max = [9]
xgb_alpha = [3]
xgb_num_parallel_tree = [3]

ada_estimator = ['fit', 'predict', None]
ada_n = [435, 440, 445]
ada_lr = [0.1, 1, 0.01]

rf_n = [220, 225, 230]
rf_max = [11, 13, 15]
rf_min_samples_leaf = [1,2,3]

gbm_n = [200]
gbm_max = [7]

gbm_loss = ['log_loss', 'exponential']
gbm_criterion = ["friedman_mse", "squared_error"]

gbm_lr = [1]
gbm_max_features = ['sqrt', 'log2']

lgbm_lr = [1]
lgbm_n_estimators = [25,50]
lgbm_max_depth = [6]
lgbm_num_leaves = [10,20,30]
import time

start_time = time.time()
models = [

    ("cB",CatBoostClassifier(), {'iterations': [200, 250],'learning_rate': [0.15],'depth': [10,15,20]}),

]

dataset = pd.read_csv("../datasets/dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, 1:dataset.shape[1]],\
                                                    dataset["label"], test_size=0.20, random_state=42)

best_models = {}
results = []

for name, model, params in models:
    print(f"Performing grid search for {name}...")
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)                                                 
    best_models[name] = grid_search.best_estimator_

    results.append({
        'Model': name,
        'Best Parameters': grid_search.best_params_,
        'Parameters Evaluated': params,
        'Best Accuracy': grid_search.best_score_
    })

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best accuracy for {name}: {grid_search.best_score_}\n")

results_dir = "../results/gridsearch"
results_df = pd.DataFrame(results)
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
excel_file_path = "grid_search_results.csv"
results_df.to_csv(os.path.join(results_dir, f"{current_datetime}_{excel_file_path}"), index=False)
end_time = time.time()
print(f"Program ran for {(end_time-start_time)/60} minutes")
print(f"Grid search results saved to {excel_file_path}")