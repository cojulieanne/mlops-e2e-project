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

test_df = pd.read_csv("data/gold/test_ml2_students_data.csv")
train_df = pd.read_csv("data/gold/train_ml2_students_data.csv")


model = RandomForestClassifier(random_state=143)

pipeline = Pipeline([('RandomUnderSampler',
                              RandomUnderSampler(random_state=143)),
                             ('RFClassifier',model)])


X_train, y_train = train_df.drop("Pass/Fail (1=Pass, 0=Fail)", axis=1), train_df["Pass/Fail (1=Pass, 0=Fail)"]
X_test, y_test = test_df.drop("Pass/Fail (1=Pass, 0=Fail)", axis=1), test_df["Pass/Fail (1=Pass, 0=Fail)"]

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}")
print(f"Test Recall: {sensitivity_score(y_test, y_pred) * 100:.2f}")
print(f"Test G-Mean: {geometric_mean_score(y_test, y_pred) * 100:.2f}")
