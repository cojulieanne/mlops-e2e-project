# MLOPS Repository

This project implements an end-to-end MLOps workflow for a student-performance classifier. It uses a bronze → silver → gold data layout, Airflow for orchestration, MLflow for tracking & registry, Evidently for drift detection, and FastAPI for serving.

## Environment Setup
- Install UV using pip. `pip install uv`

- Create a virtual environment: `uv venv .venv`

- Install dependencies: `uv add pandas numpy seaborn scikit-learn tqdm gdown logging apache-airflow` `uv pip install pre-commit`

## Documentation & Architecture

All project documentation is organized under the [`docs/`](./docs) folder:

- [`docs/dataset.md`](./docs/dataset.md) → Dataset description, size, source, preprocessing workflow, and purpose.

- [`docs/data_dictionary.md`](./docs/data_dictionary.md) → Schema describing each feature, data type, and expected values.

- [`docs/drift_plan.md`](./docs/drift_plan.md) → Plan for simulating and detecting data drift using Evidently.

- [`docs/architecture.md`](./docs/architecture.md) → System architecture description.

## Folder Structure

From the project root:
- .venv/ – virtual environment

- airflow/ – DAG definitions and configs

- data/ – bronze (raw), silver (preprocessed), gold (splits)

- docs/ – dataset, dictionary, drift plan, and architecture docs

- logs/ – Airflow and pipeline logs

- mlruns/ – MLflow tracking logs & artifacts

- models/ – saved models (e.g., model.pkl)

- notebooks/ – exploratory, preprocessing, and demo notebooks

- reports/ – evaluation outputs and drift reports

- src/ – modular pipeline code

## Testing and GitHub actions

[add here]

## Pre-commit hooks

Configured for code quality and reproducibility:

- flake8 – PEP-8 linting

- black – auto-formatting

- trailing-whitespace – clean whitespace

- hadollint – lint Dockerfiles

- yamllint – validate YAML

- ruff – linting & auto-fixes

Usage:
`pip install pre-commit`
`pre-commit install`
`pre-commit run –all-files`

# Docker and Airflow Setup

- Installed Docker Desktop by downloading the installer here: https://www.docker.com/products/docker-desktop/

- Updated Windows Subsystem for Linux by using the code: wsl --update

[add here]

# Installation dependencies for MLFlow:
mlflow = "==3.1.4"
evidently = "==0.7.11"
psycopg2-binary = "==2.9.10"

# How To Run

[add here]