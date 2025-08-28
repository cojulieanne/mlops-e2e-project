import numpy as np
import pandas as pd
import time
import pickle
import warnings
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.metrics import sensitivity_score, geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import mlflow
import mlflow.pyfunc
import joblib

warnings.filterwarnings("ignore")

class CustomMLModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow PyFunc model wrapper for your trained model."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None

    def load_context(self, context):
        """Load model artifacts from MLflow context."""
        self.model = joblib.load(context.artifacts["model"])
        if "preprocessor" in context.artifacts:
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        if "feature_names" in context.artifacts:
            with open(context.artifacts["feature_names"], 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.preprocessor:
            processed_input = self.preprocessor.transform(model_input)
        else:
            processed_input = model_input.values
        return self.model.predict(processed_input)


<<<<<<< HEAD
def training(train_x = "data/gold/X_train.csv", train_y = "data/gold/y_train.csv"):
=======
def training(train_x = "data/gold/X_train.csv" , train_y = "data/gold/y_train.csv"):
>>>>>>> origin
    X_trainval, y_trainval = pd.read_csv(train_x), pd.read_csv(train_y)

    models_dict = {
        "LogisticRegressor": LogisticRegression(penalty="l2"),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=143),
        "RandomForestClassifier": RandomForestClassifier(random_state=143),
    }

    skf = StratifiedKFold(n_splits=5)
    undersampler = {}

    total_start = time.time()

    for model_name, model in tqdm(models_dict.items()):
        val_rec_scores = []
        val_gmean_scores = []
        val_accuracy_scores = []

        for train_index, val_index in skf.split(X_trainval, y_trainval):
            X_train, X_val = X_trainval.iloc[train_index], X_trainval.iloc[val_index]
            y_train, y_val = y_trainval.iloc[train_index], y_trainval.iloc[val_index]

            start_time = time.time()

            pipeline = Pipeline(
                [
                    ("RandomUnderSampler", RandomUnderSampler(random_state=143)),
                    (model_name, model),
                ]
            )
            pipeline.fit(X_train, y_train)

            val_preds = pipeline.predict(X_val)

            val_rec_score = sensitivity_score(y_val, val_preds)
            val_gmean_score = geometric_mean_score(y_val, val_preds)
            val_accuracy_score = accuracy_score(y_val, val_preds)

            end_time = time.time()

            val_rec_scores.append(val_rec_score)
            val_gmean_scores.append(val_gmean_score)
            val_accuracy_scores.append(val_accuracy_score)

        undersampler[model_name] = {
            "ave_val_recall": np.mean(val_rec_scores) * 100,
            "ave_val_gmean_score": np.mean(val_gmean_scores) * 100,
            "ave_val_accuracy_score": np.mean(val_accuracy_scores) * 100,
            "run_time": end_time - start_time,
        }

    total_end = time.time()
    elapsed = total_end - total_start

    undersampler = pd.DataFrame(undersampler).T

    best_model_name = undersampler["ave_val_recall"].idxmax()
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


    mlflow.set_tracking_uri("http://mlflow:5000")
    with mlflow.start_run():
        if best_model_name == "RandomForestClassifier":
            mlflow.log_param("n_estimators", best_model.n_estimators if hasattr(best_model, "n_estimators") else None)
            mlflow.log_param("max_depth", best_model.max_depth)
            mlflow.log_param("random_state", best_model.random_state)
        elif best_model_name == "DecisionTreeClassifier":
            mlflow.log_param("max_depth", best_model.max_depth)
            mlflow.log_param("criterion", best_model.criterion)
            mlflow.log_param("random_state", best_model.random_state)
        elif best_model_name == "LogisticRegressor":
            mlflow.log_param("penalty", best_model.penalty)
            mlflow.log_param("solver", best_model.solver)
            mlflow.log_param("max_iter", best_model.max_iter)

        mlflow.log_metrics({
            "avg_recall": undersampler.loc[best_model_name, "ave_val_recall"],
            "avg_gmean": undersampler.loc[best_model_name, "ave_val_gmean_score"],
            "avg_accuracy": undersampler.loc[best_model_name, "ave_val_accuracy_score"],
            "elapsed_time": elapsed
        })

        mlflow.pyfunc.log_model(
            artifact_path="custom_model",
            python_model=CustomMLModel(),
            artifacts={"model": "models/model.pkl"},
            registered_model_name="BestClassifierModel"   # 👈 registers here
        )