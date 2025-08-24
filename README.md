# MLOPS Repository

## Environment Setup
-- Install UV using pip. "pip install uv" \n

-- uv venv .venv

-- uv add pandas numpy seaborn scikit-learn tqdm gdown logging apache-airflow

-- uv pip install pre-commit

## Folder Structure

--bronze
    raw data. this is where the pulled data from: https://drive.google.com/uc?id=16_IoRl6EUCevWf4_l5orzRLjlnVG-WUd is saved.

--silver
    preprocessed data. switched the labellings for the target variable

--gold
    contains test and train dataset ready for modelling.

## Scripts
-- 01.preprocessing

    added the code here to switch the values of the pass/fail column, then saved the output csv to silver

-- 02.feature engineering

    contains code for preprocessed data to split into test and train datasets, ready for model usage.

-- 03.training

    contains training code which was for the test dataset only

-- 04.evaluation

    contains evaluation code here which uses the holdout value and outputs just the accuracy.

## pre-commit hooks

Tested precommits by:

1. Install pre-commit in environment by: pip install pre-commit

2. "Pre-commit install" to run hooks automatically before committing

3. pre-commit run --all-files to check if all files are compliant


--flake8

    This was the mostly suggested hook, and I guess my most needed since i dont usually follow pep-8 :)

--black

    I found this with the help of chatgpt because when I ran flake8 i initially thought that it will fix the issues, but it did not.

    So, black does it for me which is very useful and efficient.

--trailing-whitespace

    This will be useful in tidying up my code since I noticed when revisiting my old notebooks for this homework that I do have a lot
    of unnecessary press of the "Enter" button

--hadollint

    Keeps dockerfiles clean and standardized and promotes leaner image

--yamllint

    Improves readability of my code and validates my code. Detects error like indentation, accidental presses or random characters in the code due to some accidents.

--ruff
    For autocorrection of non-compliance, similar to black and flake8.
## Model

--generated model.pkl after training


# Project Overview

-- This ML exercise was done in our ML2 subject. its a basic, direct to the point modelling challenge that was easy to understand. It allows us to check how different resampling
methods and models perform to predict the minority, in this specific problem the failed students.

# Data Repo

-- Data used for this project can be accessed here: https://drive.google.com/uc?id=16_IoRl6EUCevWf4_l5orzRLjlnVG-WUd

# Challenge while doing this homework

-- This was actually fun and was very interesting for me. This is the first time i realized why we need to save the models (in the notebook setting i dont see the point).
One challenge for me was fixing the creating the run_pipeline.py since I have to "tahi" my files to have a good flow. Also, i encountered errors in importing the scripts but was able to find the right syntax after 5-10 attempts :)

# Docker Installation

-- Installed Docker Desktop by downloading the installer here: https://www.docker.com/products/docker-desktop/

-- Updated Windows Subsystem for Linux by using the code: wsl --update

# Docker and airflow folders

-- Added docker folder and airflow folder

-- Added airflow/dags to ensure modularity, allowing independent testing of workflow tasks without affecting the core ml code

-- if this error is encountered "Virtualization support not detected" make sure to enable SVM on your BIOS

-- Verify installation with "docker --version

-- Got the docker-compose.yaml here: https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#fetching-docker-compose-yaml

-- Pulled apache airflow in bash using command: "docker pull apache/airflow:2.9.3"

-- use this code to run docker file: docker build -f Dockerfile -t 72b86e078843edb3752c88bdd311e512da2f7d46f9f8aef109f4b9f55c54984d-ml-pipeline .

-- use this code to run docker: docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  72b86e078843edb3752c88bdd311e512da2f7d46f9f8aef109f4b9f55c54984d-ml-pipeline

-- generated .env file AIRFLOW_UID then checked id value by using command id-u

-- initialized airflow by using command: docker compose up airflow-init

-- run all services using command: docker compose up -d (this allows the server to run)

username: admin
for pw: check plugins folder
### I used root user configuration because I'm encountering errors if not using user: 0:0. I know this is not the best practice prof :D


# Model Drift Detection
For this project, data drift is simulated to evaluate the robustness of the pipeline and model monitoring. For numerical features such as age, hours of sleep, and hours reviewing, drift is introduced by either multiplying values by 1.2 or adding Gaussian noise with a standard deviation equal to 10% of the feature’s variability. For categorical features like course type and relationship status, 10–15% of the values are randomly flipped to other valid categories using uniform random selection. The target variable (Pass/Fail) is intentionally left unchanged so that only the feature distributions are affected. This approach creates somewhat "realistic" scenarios where changes in demographics, lifestyle, or reporting practices alter the data distribution, allowing us to test whether the MLflow-based monitoring system can effectively detect and respond to model drift.


# Install dependencies for MLFlow:
mlflow = "==3.1.4"
evidently = "==0.7.11"
psycopg2-binary = "==2.9.10"

# Folder Structure
I set up the folder structure to keep experiments, artifacts, and reports easy to manage. The mlflow folder is where MLflow automatically tracks my runs and saves model artifacts, so everything related to experiments is organized in one folder. I also added a reports folder for outputs like drift detection results, which makes it easier to monitor changes in the data over time. This way, the workflow is organized, and anyone looking at the project can quickly find what they need.

# Dockerfile.mlflow
Created a dockerfile for mlflow.

# Docker-compose
Modified the postgres service to mlflow-postgres to avoid conflicts in the exisiting postgres service

# Composed and tested the new docker
docker-compose up -d
curl http://localhost:5000

This returned an html after running.

# Model Registration
Since the goal of the model is to maximize recall, I set up a recall > 0.8 threshold.
This is realistic since logically, we also dont want students failing so we want to detect even the slightest drift in the model.

# Test standalone pipeline
python src/run_pipeline.py
# Test Airflow DAG
airflow dags test ml_pipeline_dag 2025-08-02
# Verify MLFlow UI
curl http://localhost:5000


### Challenges for HW3

Coming from homework2, where I have a very hard time fixing docker, I guess it was the bulk of my task in homework3 as well. I have spent sleepless nights to make this project work.
The hardest challenge working with homework 3 was the import for evidently. I know it may sound funny, but I did read the documentation, tried different versions of the package, asked gpt about the versions and compatibilities, and restructured my code (I also messaged you prof), but to no avail. I got a message from 3 of my cohortmates and they have the same issue and none of us can solve it, until another cohortmate responded that he was able to solve the issue. The imports were different from the project specifications (the Required Imports: evidently.report.Report , evidently.metric_preset.DataDriftPreset).

After that it was easy. I was able to move from the issue and test each component. So happy I get to have this project working.

This experience is all new for me, and for sure I know that I learned a lot from trying.