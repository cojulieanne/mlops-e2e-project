# MLOPS Repository

## Environment Setup
-- Install UV using pip. "pip install uv" \n

-- uv venv .venv

-- uv add pandas numpy seaborn scikit-learn tqdm gdown logging

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
--flake8

    This was the mostly suggested hook, and I guess my most needed since i dont usually follow pep-8 :)

--black

    I found this with the help of chatgpt because when I ran flake8 i initially thought that it will fix the issues, but it did not.

    So, black does it for me which is very useful and efficient.

--trailing-whitespace

    This will be useful in tidying up my code since I noticed when revisiting my old notebooks for this homework that I do have a lot
    of unnecessary press of the "Enter" button

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