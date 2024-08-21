import torch
import numpy as np


# Generate a list of recommended movies for a given user
def get_recommended_movies(
    model,
    userID,  # User ID for whom recommendations are generated
    ratings,  # Ratings data
    movies,  # Movies data
    genres_split,  # One-hot encoded genres
    user_to_index,  # Mapping from user IDs to indices
    movie_to_index,  # Mapping from movie IDs to indices
    top_k,  # Number of top recommendations to return
    device,  # Device to perform computations on
    rated_movies=None,  # Optional list of already rated movies
):
    # Prepare genre information for all movies
    genres_movie = movies["genres"].str.get_dummies("|")
    movie_genres = movies[["movieId"]].join(genres_movie)
    user_index = user_to_index[userID]

    # Get all movies and those seen by the user
    all_movies = ratings["movieId"].unique()
    if rated_movies is None:
        seen_movies = ratings[ratings.userId == userID]["movieId"].unique()
    else:
        seen_movies = [movie for movie, _ in rated_movies]

    # Map movie IDs to indices
    all_movie_indices = [movie_to_index[movie] for movie in all_movies]
    seen_movie_indices = [movie_to_index[movie] for movie in seen_movies]

    # Identify movies not seen by the user
    unseen_movies = set(all_movie_indices) - set(seen_movie_indices)

    index_to_movie = {index: movie for movie, index in movie_to_index.items()}

    prediction_ratings = []
    for movie_index in unseen_movies:
        movie_tensor = torch.tensor([movie_index]).to(device)
        user_tensor = torch.tensor([user_index]).to(device)

        movie_id = index_to_movie[movie_index]
        genres_vector = (
            movie_genres[movie_genres["movieId"] == movie_id]
            .drop("movieId", axis=1)
            .values
        )

        # Handle cases where the genre vector might be empty
        if genres_vector.size > 0:
            genres_vector = genres_vector[0]
        else:
            genres_vector = np.zeros(len(genres_split.columns))

        genres_tensor = torch.tensor(genres_vector).unsqueeze(0).to(device).float()

        # Predict the rating for the unseen movie
        with torch.no_grad():
            prediction = model(user_tensor, movie_tensor, genres_tensor)
        prediction_ratings.append((movie_index, prediction.item()))

    # Sort predictions and get the top recommended movies
    prediction_ratings.sort(key=lambda x: x[1], reverse=True)
    top_movies = [movie_index for movie_index, _ in prediction_ratings[:top_k]]

    # Convert indices back to movie IDs and titles
    top_movies_movieId = [index_to_movie[movie_index] for movie_index in top_movies]
    top_movies_titles = [
        movies[movies.movieId == movieId]["title"].values[0]
        for movieId in top_movies_movieId
    ]

    return top_movies_movieId, top_movies_titles
