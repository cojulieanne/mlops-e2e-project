import pandas as pd

def preprocess_data():
    df = pd.read_csv("data/bronze/ml2_student_performance.csv")
    df["Pass/Fail (1=Pass, 0=Fail)"] = df["Pass/Fail (1=Pass, 0=Fail)"].apply(
        lambda x: 1 if x == 0 else 0
    )
    df.to_csv("data/silver/preprocessed_ml2_student_performance.csv", index=False)
