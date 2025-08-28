import numpy as np
import pandas as pd
import time
import pickle
import warnings
from tqdm import tqdm
import sklearn

import sys
from pathlib import Path

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# import xgboost
# import lightgbm
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, auc, precision_recall_curve
from sklearn.exceptions import NotFittedError

import imblearn
from imblearn.metrics import sensitivity_score, geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

import joblib
# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from evaluation.evaluation import evaluate_single_model

import logging

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# ------------------------------ Config ------------------------------

MLFLOW_TRACKING_URI =  "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return here

PROJECT_ROOT = project_root()
print(PROJECT_ROOT)

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


def get_default_binary_models(cv = None):
    """
    Trains and evaluates a set of default binary classification models.

    Returns:
        dict: model_name → metrics
    """
    X_train = pd.read_csv(PROJECT_ROOT/"data/gold/X_train.csv")
    y_train = pd.read_csv(PROJECT_ROOT/"data/gold/y_train.csv")
    X_test = pd.read_csv(PROJECT_ROOT/"data/gold/X_test.csv")
    y_test = pd.read_csv(PROJECT_ROOT/"data/gold/y_test.csv")

    client = MlflowClient()
    mlflow.set_experiment("ML Experiments") 

    models = {
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "LogReg": LogisticRegression(penalty="l2"),
        "RandomForest": RandomForestClassifier(random_state=143),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=143),
        "GradBoost": GradientBoostingClassifier(random_state=143),
        # "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        # "LightGBM": LGBMClassifier(verbose = -1)
    }

    resamplers = {
        "NoReSampling": None,
        "RandomUnderSampler": RandomUnderSampler(random_state=143),
        "RandomOverSampler": RandomOverSampler(random_state = 143),
        "SMOTE": SMOTE(random_state=143, sampling_strategy = 'auto'),
    }

    print(f"Loaded {len(models)} binary classifiers. Evaluating...")

    results = {}    
    with mlflow.start_run(run_name="default_classifiers") as parent_run:
        mlflow.set_tags({
            "suite": "default_binary_models"
        })

        for model_name, model in models.items():
            mlflow.set_tag("base_model", model_name)
            for sampling_name, sampling in resamplers.items():
                mlflow.set_tag("resampling", sampling_name)
                pipe_name = f"{sampling_name}_{model_name}"
                print(pipe_name)

                if sampling:
                    pipeline = Pipeline(
                            [
                                (sampling_name, sampling),
                                (model_name, model),
                            ]
                        )
                else:
                    pipeline = model
                    pipe_name = model_name


                try:
                    model, metrics, cv_metrics = evaluate_single_model(pipeline, cv = cv)
                    results[pipe_name] = metrics | cv_metrics
                    print(f"{pipe_name} → {metrics}")
                    print(f"CV → {cv_metrics}\n")

                    with mlflow.start_run(run_name=pipe_name, nested=True):
                        mlflow.log_param("model", model_name)
                        mlflow.log_param("sampler", sampling_name if sampling else "None")

                        try:
                            if sampling: 
                                for k, v in model[0].get_params().items():
                                    if isinstance(v, (int, float, str, bool)):
                                        mlflow.log_param(f"sampler__{k}", v)
                                
                                for k, v in model[1].get_params().items():
                                    if isinstance(v, (int, float, str, bool)):
                                        mlflow.log_param(f"model__{k}", v)
                            else:
                                for k, v in model.get_params().items():
                                    if isinstance(v, (int, float, str, bool)):
                                        mlflow.log_param(f"model__{k}", v)
                        except Exception:
                            pass

                        for k, v in metrics.items():
                            if isinstance(v, (int, float, np.floating)):
                                mlflow.log_metric(k, float(v))
                        if cv and cv_metrics:
                            for k, v in cv_metrics.items():
                                if isinstance(v, (int, float, np.floating)):
                                    mlflow.log_metric(k, float(v))


                except Exception as e:
                    logging.exception("%s failed", pipe_name)
                    results[pipe_name] = "Failed"

        results_df = pd.DataFrame(results).T  # transpose if models are keys
        results_df = results_df.sort_values(by = 'cv_geometric_mean', ascending=False)
        out_csv = PROJECT_ROOT/"reports"/"default_binary_models_results.csv"
        print(out_csv)
        results_df.to_csv(out_csv)
        print("Saved results to default_binary_models_results.csv")

        #mlflow.log_artifact(out_csv, artifact_path="reports")

        
        results_df['exclude'] = results_df.apply(lambda row: (row == 0).sum(), axis=1) > 0
        results_df[results_df['exclude'] == 0].sort_values(by = 'cv_geometric_mean', ascending=False)
        print(f"Best Default Model: {results_df.index[0]}")
        print(results_df.head(1).to_dict())

        if not results_df.empty:
            best_name = results_df.index[0]
            mlflow.set_tag("best_model", best_name)
            metrics = results[best_name]

            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating)):
                    mlflow.log_metric(k, float(v))

            exp_id = mlflow.active_run().info.experiment_id

            best_run = client.search_runs(
                        experiment_ids=exp_id,
                        filter_string=f"tags.mlflow.runName = '{best_name}'",
                        order_by=["attributes.start_time DESC"],
                        max_results=1,
                    )
            
            params = best_run[0].data.params
            sampler_name = params['sampler']
            model_name = params['model']
            best_resampler = resamplers[sampler_name]
            best_model = models[model_name]

            params = {k.split("__", 1)[1]: _coerce(v)
                        for k, v in params.items() if 'model__' in k}

            best_model.set_params(**params)

            if best_resampler:
                best_pipe = Pipeline(
                    [
                        (sampler_name, best_resampler),
                        (model_name, best_model),
                    ]
                )
            else:
                best_pipe = best_model

            best_pipe, best_results, __ = evaluate_single_model(best_pipe)

            print(best_results)

            X_ex = X_test.iloc[:5]
            yhat_ex = best_pipe.predict_proba(X_ex)
            signature = infer_signature(X_ex, yhat_ex)
            print(signature)

            joblib.dump(best_pipe, PROJECT_ROOT/"models"/f"{best_name}.joblib")
            mlflow.pyfunc.log_model(
                name=best_name,
                python_model=CustomMLModel(),
                artifacts={"model":  f"{PROJECT_ROOT}/models/{best_name}.joblib"},                # your mlflow.pyfunc.PythonModel
                input_example=X_ex,
                signature=signature,
                pip_requirements=[
                    f"numpy=={np.__version__}",
                    f"pandas=={pd.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"imbalanced-learn=={imblearn.__version__}"                        # add any others you import
                ],
                registered_model_name=f"BestClassifierModel_{best_name}"   # 👈 registers here
            )
        

    return results


def _coerce(v):
    if isinstance(v, str):
        if v == "True":  return True
        if v == "False": return False
        try:
            # ints first, then floats
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
            return float(v)
        except Exception:
            return v
    return v
