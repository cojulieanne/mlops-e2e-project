import gdown


def load():
    file_id = "16_IoRl6EUCevWf4_l5orzRLjlnVG-WUd"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "data/bronze/ml2_student_performance.csv"

    gdown.download(url, output, quiet=False)

def main():
    load()

if __name__ == "__main__":
    main()