import pickle
import json
import warnings
from imblearn.metrics import sensitivity_score, geometric_mean_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


def evaluation(xtrain, ytrain, xtest, ytest):
    with open("models/model.pkl", "rb") as f:
        pipeline = pickle.load(f)

    pipeline.fit(xtrain, ytrain)

    y_pred = pipeline.predict(xtest)

    accuracy = accuracy_score(ytest, y_pred) * 100
    recall = sensitivity_score(ytest, y_pred) * 100
    gmean = geometric_mean_score(ytest, y_pred) * 100

    results = {
        "accuracy": round(accuracy, 2),
        "recall": round(recall, 2),
        "gmean": round(gmean, 2)
    }

    with open("reports/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)