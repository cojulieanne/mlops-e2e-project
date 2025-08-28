import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_data():
    df = pd.read_csv("data/bronze/ml2_student_performance.csv")

    df["Pass/Fail (1=Pass, 0=Fail)"] = df["Pass/Fail (1=Pass, 0=Fail)"].apply(
        lambda x: 1 if x == 0 else 0
    )

    gold_path = "data/gold/preprocessed_ml2_student_performance.csv"
    df.to_csv(gold_path, index=False)


    target_col = "Pass/Fail (1=Pass, 0=Fail)"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    numerical_cols = ["Age", "Hours of Sleep", "Hours Reviewing"]
    categorical_cols = [
        "Course (STEM=1, Non-STEM=0)",
        "In a Relationship (1=Yes, 0=No, 0.5=It's complicated)",
    ]
    
    X_train.to_csv("data/gold/X_train.csv", index=False)
    X_test.to_csv("data/gold/X_test.csv", index=False)
    y_train.to_csv("data/gold/y_train.csv", index=False)
    y_test.to_csv("data/gold/y_test.csv", index=False)

    X_train.to_csv("data/gold/X_train.csv", index=False)
    X_test.to_csv("data/gold/X_test.csv", index=False)
    y_train.to_csv("data/gold/y_train.csv", index=False)
    y_test.to_csv("data/gold/y_test.csv", index=False)

    #Drift
    def create_drifted_copy(X_data):
        drifted = X_data.copy()

        for col in numerical_cols:
            if np.random.rand() > 0.5:
                drifted[col] = drifted[col] * 1.2
            else:
                std = X_data[col].std()
                drifted[col] = drifted[col] + np.random.normal(
                    0, 0.1 * std, size=len(X_data)
                )

        for col in categorical_cols:
            unique_vals = X_data[col].unique()
            n = len(drifted)
            flip_mask = np.random.rand(n) < np.random.uniform(0.1, 0.15)
            random_choices = np.random.choice(unique_vals, size=n)
            drifted.loc[flip_mask, col] = random_choices[flip_mask]

        return drifted

    X_train_drifted = create_drifted_copy(X_train)
    X_test_drifted = create_drifted_copy(X_test)

    drifted_train = X_train_drifted.copy()
    drifted_train[target_col] = y_train.values
    drifted_test = X_test_drifted.copy()
    drifted_test[target_col] = y_test.values

    drifted_train.to_csv("data/gold/drifted_train.csv", index=False)
    drifted_test.to_csv("data/gold/drifted_test.csv", index=False)
    
<<<<<<< HEAD

    # return (X_train,
    #         X_test,
    #         y_train,
    #         y_test,
    #         X_train_drifted,
    #         X_test_drifted)
=======
>>>>>>> origin
