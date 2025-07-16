import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import time
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")

# metrics
from imblearn.metrics import sensitivity_score, geometric_mean_score

# resampling methods
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# pipeline
from imblearn.pipeline import Pipeline

df = pd.read_csv("data/gold/train_ml2_students_data.csv")

X_trainval, y_trainval = df.drop("Pass/Fail (1=Pass, 0=Fail)", axis=1), df["Pass/Fail (1=Pass, 0=Fail)"]

models_dict = {
    'LogisticRegressor': LogisticRegression(penalty='l2'),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=143),
    'RandomForestClassifier': RandomForestClassifier(random_state=143)
}

skf = StratifiedKFold(n_splits=5)

undersampler = {}

# log start time
total_start = time.time()

for model_name, model in tqdm(models_dict.items()):
    val_rec_scores = []
    val_gmean_scores = []
    val_accuracy_scores = []

    for train_index, val_index in skf.split(X_trainval, y_trainval):
        X_train, X_val = X_trainval.iloc[train_index], X_trainval.iloc[val_index]
        y_train, y_val = y_trainval.iloc[train_index], y_trainval.iloc[val_index]

        start_time = time.time() # for logging run times

        pipeline = Pipeline([('RandomUnderSampler',
                              RandomUnderSampler(random_state=143)),
                             (model_name, model)])
        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict(X_train)
        val_preds = pipeline.predict(X_val)

        val_rec_score = sensitivity_score(y_val, val_preds)
        val_gmean_score = geometric_mean_score(y_val, val_preds)
        val_accuracy_score = accuracy_score(y_val, val_preds)
        
        end_time = time.time() # for logging run times

        val_rec_scores.append(val_rec_score)
        val_gmean_scores.append(val_gmean_score)
        val_accuracy_scores.append(val_accuracy_score)

    undersampler[model_name] = {
        'ave_val_recall':np.mean(val_rec_scores) * 100,
        'ave_val_gmean_score':np.mean(val_gmean_scores) * 100,
        'ave_val_accuracy_score':np.mean(val_accuracy_scores) * 100,
        'run_time': end_time - start_time
    }

# log end time
total_end = time.time()

elapsed = total_end - total_start
# print(f"Report Generated in {elapsed:.2f} seconds")
undersampler = pd.DataFrame(undersampler).T
print(undersampler)
