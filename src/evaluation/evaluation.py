import pickle
import json
import warnings
import pandas as pd
from imblearn.metrics import sensitivity_score, geometric_mean_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, f1_score

warnings.filterwarnings("ignore")


def evaluation():
    with open("models/model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    xtrain=pd.read_csv("data/gold/X_train.csv")
    ytrain=pd.read_csv("data/gold/y_train.csv")
    xtest=pd.read_csv("data/gold/X_test.csv")
    ytest=pd.read_csv("data/gold/y_test.csv")
    pipeline.fit(xtrain, ytrain)

    y_pred = pipeline.predict(xtest)
    y_proba = pipeline.predict_proba(xtest)[:, 1]

    accuracy = accuracy_score(ytest, y_pred) * 100
    recall = recall_score(ytest, y_pred) * 100
    precision = precision_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)
    gmean = geometric_mean_score(ytest, y_pred) * 100
    roc_auc = roc_auc_score(ytest, y_proba)
    pr_auc = average_precision_score(ytest, y_proba)

    results = {
        "accuracy": round(accuracy, 2),
        "recall": round(recall, 2),
        "precision": round(precision, 2),
        "f1_score": round(f1, 2),
        "gmean": round(gmean, 2),
        "roc_auc": round(roc_auc, 2),
        "pr_auc": round(pr_auc,2 )
    }

    with open("reports/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

 