import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

def split_data(link):
    df = pd.read_csv(link)

    X, y = df.drop("Pass/Fail (1=Pass, 0=Fail)", axis=1), df["Pass/Fail (1=Pass, 0=Fail)"]

    (X_trainval, X_holdout, y_trainval, y_holdout) = train_test_split(
        X, y, random_state=143, test_size=0.25, stratify=y
    )

    train_df = X_trainval.copy()
    train_df["Pass/Fail (1=Pass, 0=Fail)"] = y_trainval

    test_df = X_holdout.copy()
    test_df["Pass/Fail (1=Pass, 0=Fail)"] = y_holdout


    train_df.to_csv("data/gold/train_ml2_students_data.csv", index=False)
    test_df.to_csv("data/gold/test_ml2_students_data.csv", index=False)

    

    return train_df, test_df