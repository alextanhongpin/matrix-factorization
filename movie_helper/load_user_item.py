import os
import pandas as pd

file_dir = "ml-100k"
file_path = os.path.join(file_dir, "u.data")

names = ["user_id", "item_id", "rating", "timestamp"]


def load_user_item():
    return pd.read_csv(file_path, sep="\t", names=names)
