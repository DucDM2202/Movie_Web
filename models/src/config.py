# Training parameters
lr = 1e-3
wd = 1e-5
bs = 2000
n_epochs = 100
patience = 10

# Model parameters
n_factors = 150
hidden_size = [500, 500, 500]
embedding_dropout = 0.05
dropouts = [0.5, 0.5, 0.25]

# Recommendation parameters
top_k = 10
