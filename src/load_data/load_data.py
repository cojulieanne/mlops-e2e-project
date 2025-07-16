import gdown
import pandas as pd

def load_data():
    file_id = "16_IoRl6EUCevWf4_l5orzRLjlnVG-WUd"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "data/bronze/ml2_student_performance.csv"

    gdown.download(url, output, quiet=False)
