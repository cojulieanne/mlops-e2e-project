import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import time
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

# metrics
from imblearn.metrics import sensitivity_score, geometric_mean_score

#resampling methods
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# pipeline
from imblearn.pipeline import Pipeline

df = pd.read_csv('data/silver/preprocessed_ml2_student_performance.csv')

X, y = df.drop('Pass/Fail (1=Pass, 0=Fail)', axis=1), df['Pass/Fail (1=Pass, 0=Fail)']

(X_trainval, X_holdout, y_trainval, y_holdout) = train_test_split(X, y,
                                                                  random_state=143,
                                                                  test_size=0.25,
                                                                  stratify=y)

models_dict = {
    'RandomForestClassifier': RandomForestClassifier(random_state=143)
}

skf = StratifiedKFold(n_splits=5)

res = {}

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

        # fit
        model.fit(X_train, y_train)

        #predict
        val_preds = model.predict(X_holdout)

        val_rec_score = sensitivity_score(y_holdout, val_preds)
        val_gmean_score = geometric_mean_score(y_holdout, val_preds)
        val_accuracy_score = accuracy_score(y_holdout, val_preds)

        end_time = time.time() # for logging run times

        val_rec_scores.append(val_rec_score)
        val_gmean_scores.append(val_gmean_score)
        val_accuracy_scores.append(val_accuracy_score)

    res[model_name] = {
        'ave_val_recall':np.mean(val_rec_scores) * 100,
        'ave_val_gmean_score':np.mean(val_gmean_scores) * 100,
        'ave_val_accuracy_score':np.mean(val_accuracy_scores) * 100,
        'run_time': end_time - start_time
    }

# log end time
total_end = time.time()

elapsed = total_end - total_start
#print(f"Report Generated in {elapsed:.2f} seconds")
res = pd.DataFrame(res).T
print(res)