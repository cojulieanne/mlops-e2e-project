# 72b86e078843edb3752c88bdd311e512da2f7d46f9f8aef109f4b9f55c54984d_Homework1

## Environment Setup
-- Install UV using pip. "pip install uv"
-- uv venv .venv
-- uv add pandas numpy seaborn scikit-learn tqdm gdown
-- added uv.lock and .venv/ in .gitignore
-- uv pip install pre-commit

## Folder Structure
--bronze
    raw data. this is where the pulled data from: https://drive.google.com/uc?id=16_IoRl6EUCevWf4_l5orzRLjlnVG-WUd is saved.
--silver
    preprocessed data. switched the labellings for the target variable
--gold
    I am not sure prof what to put here to be honest given that my model is very simple.

## Models
-- 01.preprocessing
    added the code here to switch the values of the pass/fail column, then saved the output csv to silver
-- 02.feature engineering
    dataset was pretty direct so no need to add feature engineering code here, but i put it to show structure in the models folder
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