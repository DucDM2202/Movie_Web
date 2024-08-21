import numpy as np
import pandas as pd


# Create a dataset with encoded users, movies, and genres
def create_dataset_with_genres(ratings, genres_split, top=None):
    if top is not None:
        ratings.groupby("userId")["rating"].count()

    # Map unique users to indices
    unique_users = ratings.userId.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)

    # Map unique movies to indices
    unique_movies = ratings.movieId.unique()
    movie_to_index = {old: new for new, old in enumerate(unique_movies)}
    new_movies = ratings.movieId.map(movie_to_index)

    # Get number of unique users and movies
    n_users = unique_users.shape[0]
    n_movies = unique_movies.shape[0]

    # Create feature matrix X and target vector y
    X = pd.DataFrame({"user_id": new_users, "movie_id": new_movies})
    y = ratings["rating"].astype(np.float32)

    # Add genre information to feature matrix
    X = pd.concat([X, genres_split], axis=1)

    # Return dataset information and mappings
    return (
        (n_users, n_movies, genres_split.shape[1]),
        (X, y),
        (user_to_index, movie_to_index),
    )
