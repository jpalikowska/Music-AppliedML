import os
import pandas as pd

def load_or_download_csv(file_name, url, column_names=None):
    if os.path.exists(file_name):
        print(f"Loading from local `{file_name}`...")
        return pd.read_csv(file_name, index_col=0)
    else:
        print(f"Downloading from `{url}`...")
        df = pd.read_csv(url, names=column_names)
        df.to_csv(file_name)
        print("Saved to local file.")
        return df