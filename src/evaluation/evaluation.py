import pickle
from imblearn.metrics import sensitivity_score, geometric_mean_score
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


def evaluation(xtrain, ytrain, xtest, ytest):


    with open("models/model.pkl", "rb") as f:
        pipeline = pickle.load(f)

    X_train, y_train = xtrain, ytrain
    X_test, y_test = xtest, ytest

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    recall = sensitivity_score(y_test, y_pred) * 100
    gmean = geometric_mean_score(y_test, y_pred) * 100

    with open("reports/metrics.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Test Recall: {recall:.2f}%\n")
        f.write(f"Test G-Mean: {gmean:.2f}%\n")
