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