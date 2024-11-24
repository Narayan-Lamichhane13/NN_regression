# main.py

import torch
import torch.nn as nn
import torch.optim as optim

# Since we're not performing hyperparameter tuning right now, you can comment this out or adjust it for classification
# print("Starting hyperparameter tuning...")
# import hyperparameter_tuning

# If you have best parameters from hyperparameter tuning for classification, you can load them
# For simplicity, let's define some hyperparameters directly
best_params = {
    'hidden_sizes': [64, 32, 16],
    'dropout_rate': 0.5,
    'learning_rate': 0.001
}

# Assign 'hidden_sizes' and 'dropout_rate' before using them
hidden_sizes = best_params['hidden_sizes']
dropout_rate = best_params['dropout_rate']

# Import necessary modules
from data_preparation import load_and_preprocess_data
from model import ClassificationModel  # Use the classification model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data('phishing_dataset.csv')
input_size = X_train.shape[1]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Create DataLoader
from torch.utils.data import TensorDataset, DataLoader

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model, Loss Function, Optimizer
model = ClassificationModel(
    input_size,
    hidden_sizes=hidden_sizes,
    dropout_rate=dropout_rate
).to(device)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

# Training Loop
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    # Calculate average training loss
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())
        
        # Calculate validation accuracy
        val_preds = (val_outputs.cpu().numpy() > 0.5).astype(int)
        val_accuracy = (val_preds == y_val_tensor.cpu().numpy()).mean()
        
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_loss.item():.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'classification_model.pth')

# Evaluate the model
print("Evaluating the model on the test set...")
# import evaluate  # Uncomment if evaluate.py exists
