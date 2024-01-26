import os
import pandas as pd

file_dir = "ml-100k"
file_path = os.path.join(file_dir, "u.user")
names = ["user_id", "age", "gender", "occupation", "zip_code"]


def load_user():
    return pd.read_csv(file_path, sep="|", names=names)
