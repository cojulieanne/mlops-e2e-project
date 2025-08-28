import pickle
import json
import warnings
import joblib
import pandas as pd
from imblearn.metrics import sensitivity_score, geometric_mean_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import numpy as np
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, auc
from sklearn.exceptions import NotFittedError
warnings.filterwarnings("ignore")
from pathlib import Path

def project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return here

PROJECT_ROOT = project_root()
print(PROJECT_ROOT)

def evaluation():
    xtrain = pd.read_csv(PROJECT_ROOT/"data/gold/X_train.csv")
    ytrain = pd.read_csv(PROJECT_ROOT/"data/gold/y_train.csv")
    xtest = pd.read_csv(PROJECT_ROOT/"data/gold/X_test.csv")
    ytest = pd.read_csv(PROJECT_ROOT/"data/gold/y_test.csv")
    pipeline = joblib.load(PROJECT_ROOT/"models/RandomUnderSampler_RandomForest.joblib")

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

def evaluate_single_model(model, cv = None):
    """
    Trains and evaluates a single model on binary classification metrics.

    Returns:
        dict: {accuracy, precision, recall, f1_score, roc_auc}
    """
    X_train = pd.read_csv(PROJECT_ROOT/"data/gold/X_train.csv")
    y_train = pd.read_csv(PROJECT_ROOT/"data/gold/y_train.csv")
    X_test = pd.read_csv(PROJECT_ROOT/"data/gold/X_test.csv")
    y_test = pd.read_csv(PROJECT_ROOT/"data/gold/y_test.csv")
    if cv:
        cv_results = cv_binary_metrics(model, cv)

    else:
        cv_results = None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # event_rate = y_test.sum()/len(y_test)
    positive_rate = y_pred.sum()/len(y_pred)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, NotFittedError):
        y_proba = None

    test_results = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else "N/A",
        "pr_auc": round(average_precision_score(y_test, y_proba), 4) if y_proba is not None else "N/A",
        "gmean_score": round(geometric_mean_score(y_test, y_pred), 4),
        # "event_rate": round(event_rate, 4),
        "positive_rate": round(positive_rate, 4)       
    }

    return model, test_results, cv_results


def cv_binary_metrics(estimator, cv):
    X = pd.read_csv(PROJECT_ROOT/"data/gold/X_train.csv")
    y = pd.read_csv(PROJECT_ROOT/"data/gold/y_train.csv")
    acc_list, prauc_list = [], []
    pr_list, rec_list, rocauc_list, f1_list, gmean_list = [], [], [], [], []

    skf = StratifiedKFold(n_splits=cv)

    for tr, va in skf.split(X, y):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        est = estimator
        est.fit(X_tr, y_tr)

        y_pred = est.predict(X_va)
        if hasattr(est, "predict_proba"):
            y_score = est.predict_proba(X_va)[:, 1]
        elif hasattr(est, "decision_function"):
            y_score = est.decision_function(X_va)
        else:
            y_score = y_pred  # fallback

        acc_list.append(accuracy_score(y_va, y_pred))
        pr_list.append(precision_score(y_va, y_pred))
        rec_list.append(recall_score(y_va, y_pred))
        f1_list.append(f1_score(y_va, y_pred))
        gmean_list.append(geometric_mean_score(y_va, y_pred, average="binary"))

        prauc_list.append(average_precision_score(y_va, y_score))
        rocauc_list.append(roc_auc_score(y_va, y_score))

        try:
            rocauc_list.append(roc_auc_score(y_va, y_score))
        except ValueError:
            rocauc_list.append(np.nan)

    return {
        "cv_precision": float(np.mean(pr_list)),
        "cv_accuracy": float(np.mean(acc_list)),
        "cv_prauc": float(np.mean(prauc_list)),
        "cv_recall": float(np.mean(rec_list)),
        "cv_rocauc": float(np.nanmean(rocauc_list)),
        "cv_f1": float(np.mean(f1_list)),
        "cv_geometric_mean": float(np.mean(gmean_list)),
    }