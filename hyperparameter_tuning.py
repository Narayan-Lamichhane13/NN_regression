# hyperparameter_tuning.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid, KFold
from data_preparation import load_and_preprocess_data
from model import AdvancedRegressionModel

# Hyperparameters grid
param_grid = {
    'hidden_sizes': [[128, 64, 32], [64, 32, 16], [32, 16, 8]],
    'dropout_rate': [0.3, 0.5, 0.7],
    'learning_rate': [0.01, 0.001, 0.0001],
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Preparation
X_train_full, X_val_full, X_test, y_train_full, y_val_full, y_test = load_and_preprocess_data('data.csv')
input_size = X_train_full.shape[1]

# Combine training and validation sets for cross-validation
X_full = np.vstack((X_train_full, X_val_full))
y_full = np.concatenate((y_train_full, y_val_full))

# Convert to PyTorch tensors
X_full_tensor = torch.tensor(X_full, dtype=torch.float32).to(device)
y_full_tensor = torch.tensor(y_full, dtype=torch.float32).unsqueeze(1).to(device)

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

best_params = None
best_score = float('inf')

for params in ParameterGrid(param_grid):
    fold_losses = []
    print(f'Testing parameters: {params}')
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_full)):
        X_train_fold = X_full_tensor[train_idx]
        y_train_fold = y_full_tensor[train_idx]
        X_val_fold = X_full_tensor[val_idx]
        y_val_fold = y_full_tensor[val_idx]
        
        # Create model
        model = AdvancedRegressionModel(input_size, params['hidden_sizes'], params['dropout_rate']).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            # Forward pass
            outputs = model(X_train_fold)
            loss = criterion(outputs, y_train_fold)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            val_loss = criterion(val_outputs, y_val_fold)
            fold_losses.append(val_loss.item())
    avg_loss = np.mean(fold_losses)
    print(f'Average validation loss: {avg_loss:.4f}')
    if avg_loss < best_score:
        best_score = avg_loss
        best_params = params

print(f'Best Parameters: {best_params}')
print(f'Best Validation Loss: {best_score:.4f}')

# Save the best parameters
import json
with open('best_params.json', 'w') as f:
    json.dump(best_params, f)
