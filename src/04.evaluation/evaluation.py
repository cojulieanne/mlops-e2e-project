import pandas as pd
import pickle
from imblearn.metrics import sensitivity_score, geometric_mean_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


test_df = pd.read_csv("data/gold/test_ml2_students_data.csv")
train_df = pd.read_csv("data/gold/train_ml2_students_data.csv")


with open('models/model.pkl', 'rb') as f:
    pipeline = pickle.load(f)


X_train, y_train = train_df.drop("Pass/Fail (1=Pass, 0=Fail)", axis=1), train_df["Pass/Fail (1=Pass, 0=Fail)"]
X_test, y_test = test_df.drop("Pass/Fail (1=Pass, 0=Fail)", axis=1), test_df["Pass/Fail (1=Pass, 0=Fail)"]

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
recall = sensitivity_score(y_test, y_pred) * 100
gmean = geometric_mean_score(y_test, y_pred) * 100

with open('reports/metrics.txt', 'w') as f:
    f.write(f"Test Accuracy: {accuracy:.2f}%\n")
    f.write(f"Test Recall: {recall:.2f}%\n")
    f.write(f"Test G-Mean: {gmean:.2f}%\n")