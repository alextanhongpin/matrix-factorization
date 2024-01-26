import os
import pandas as pd

file_dir = "ml-100k"
file_path = os.path.join(file_dir, "u.item")

names = [
    "movie_id",
    "movie_title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def load_movies():
    return pd.read_csv(file_path, sep="|", names=names, encoding="iso-8859-1")
