import torch
import pickle
import pandas as pd
from data import create_dataset_with_genres
from model import EmbeddingNet
from train import train_model, update_embedding_layers, incremental_learning
from recommend import get_recommended_movies
from eval import eval_model
from config import *

# inputs = 1 to train model and recommend movies for a user
# inputs = 2 to incremental learning and recommend movies for a new user
inputs = 2

# Load data
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")
ratings = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
ratings = ratings.drop(["timestamp"], axis=1)
genres_split = ratings["genres"].str.get_dummies("|")

# Create dataset
(n_users, n_movies, n_genres), (X, y), (user_to_index, movie_to_index) = (
    create_dataset_with_genres(ratings, genres_split)
)

# Initialize model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = EmbeddingNet(
    n_users,
    n_movies,
    n_genres,
    n_factors=n_factors,
    hidden=hidden_size,
    embedding_dropout=embedding_dropout,
    dropouts=dropouts,
)
net.to(device)

minmax = ratings.rating.min(), ratings.rating.max()

# Train model and recommend movies for a user
if inputs == 1:
    # Train model
    net, history, lr_history = train_model(
        net, X, y, n_epochs, lr, wd, bs, patience, minmax, device
    )

    final_loss = eval_model(net, X, y, bs, minmax, device)

    # Save best model weights
    with open("checkpoints/best.weights", "wb") as file:
        pickle.dump(net.state_dict(), file)

    # Recommend movies for a user
    userID = 1
    top_movies_movieId, top_movies_titles = get_recommended_movies(
        net,
        userID,
        ratings,
        movies,
        genres_split,
        user_to_index,
        movie_to_index,
        top_k,
        device,
    )

    # Display recommended movies
    for movieId, movieTitle in zip(top_movies_movieId, top_movies_titles):
        print(f"{movieId}: {movieTitle}")

# Incremental learning and recommend movies for a new user
if inputs == 2:
    # Load best model weights
    with open("checkpoints/best.weights", "rb") as dbfile:
        best_weights = pickle.load(dbfile)
        net.load_state_dict(best_weights)

    userID = 123456789
    rated_movies = [[31, 4.5], [1129, 1.0], [6269, 3.0]]

    # Update embedding layers
    user_to_index, movie_to_index = update_embedding_layers(
        net, userID, rated_movies, user_to_index, movie_to_index
    )

    # Incremental learning
    net = incremental_learning(
        net,
        X,
        y,
        userID,
        rated_movies,
        user_to_index,
        movie_to_index,
        movies,
        genres_split,
        lr,
        wd,
        bs,
        minmax,
        device,
    )

    # Recommend movies for a new user
    top_movies_movieId, top_movies_titles = get_recommended_movies(
        net,
        userID,
        ratings,
        movies,
        genres_split,
        user_to_index,
        movie_to_index,
        top_k,
        device,
        rated_movies,
    )

    # Display recommended movies
    for movieId, movieTitle in zip(top_movies_movieId, top_movies_titles):
        print(f"{movieId}: {movieTitle}")
