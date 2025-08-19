import numpy as np
import pandas as pd

import time
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")
from imblearn.metrics import sensitivity_score, geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle


def training(train_x, train_y):
    X_trainval, y_trainval = train_x, train_y

    models_dict = {
        "LogisticRegressor": LogisticRegression(penalty="l2"),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=143),
        "RandomForestClassifier": RandomForestClassifier(random_state=143),
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

            start_time = time.time()  # for logging run times

            pipeline = Pipeline(
                [
                    ("RandomUnderSampler", RandomUnderSampler(random_state=143)),
                    (model_name, model),
                ]
            )
            pipeline.fit(X_train, y_train)

            train_preds = pipeline.predict(X_train)
            val_preds = pipeline.predict(X_val)

            val_rec_score = sensitivity_score(y_val, val_preds)
            val_gmean_score = geometric_mean_score(y_val, val_preds)
            val_accuracy_score = accuracy_score(y_val, val_preds)

            end_time = time.time()  # for logging run times

            val_rec_scores.append(val_rec_score)
            val_gmean_scores.append(val_gmean_score)
            val_accuracy_scores.append(val_accuracy_score)

        undersampler[model_name] = {
            "ave_val_recall": np.mean(val_rec_scores) * 100,
            "ave_val_gmean_score": np.mean(val_gmean_scores) * 100,
            "ave_val_accuracy_score": np.mean(val_accuracy_scores) * 100,
            "run_time": end_time - start_time,
        }

    # log end time
    total_end = time.time()

    elapsed = total_end - total_start
    # print(f"Report Generated in {elapsed:.2f} seconds")
    undersampler = pd.DataFrame(undersampler).T
    # print(undersampler)

    best_model_name = undersampler["ave_val_recall"].idxmax()
    # print(f"\nBest Model based on Average Validation recall: {best_model_name}")
    best_model = models_dict[best_model_name]

    best_pipeline = Pipeline(
        [
            ("RandomUnderSampler", RandomUnderSampler(random_state=143)),
            ("Classifier", best_model),
        ]
    )

    best_pipeline.fit(X_trainval, y_trainval)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)
