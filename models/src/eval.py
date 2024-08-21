import torch
import numpy as np
from sklearn.model_selection import train_test_split
from iterator import batches_train


# Evaluate the model performance on validation data
def eval_model(net, X, y, bs, minmax, device):

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    datasets = {"train": (X_train, y_train), "val": (X_valid, y_valid)}

    ground_truth, predictions = [], []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate through validation batches
        for batch in batches_train(*datasets["val"], shuffle=False, bs=bs):
            x_batch, y_batch = [b.to(device) for b in batch]
            user_ids = x_batch[:, 0]
            movie_ids = x_batch[:, 1]
            genres = x_batch[:, 2:]

            # Generate predictions
            outputs = net(user_ids, movie_ids, genres, minmax)
            ground_truth.extend(y_batch.tolist())
            predictions.extend(outputs.tolist())

    # Convert ground truth and predictions to numpy arrays
    ground_truth = np.asarray(ground_truth).ravel()
    predictions = np.asarray(predictions).ravel()

    # Calculate and return RMSE as the final loss
    final_loss = np.sqrt(np.mean((np.array(predictions) - np.array(ground_truth)) ** 2))
    return final_loss
