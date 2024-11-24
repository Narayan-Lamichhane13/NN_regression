# evaluate.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from model import ClassificationModel
from data_preparation import load_and_preprocess_data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Preparation
_, _, X_test, _, _, y_test = load_and_preprocess_data('phishing_dataset.csv')
input_size = X_test.shape[1]

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Load the trained model
model = ClassificationModel(input_size, hidden_sizes, dropout_rate).to(device)
model.load_state_dict(torch.load('classification_model.pth'))
model.eval()

# Prediction
with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
    probabilities = outputs.cpu().numpy().flatten()

# Calculate Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f'Accuracy on Test Set: {accuracy:.4f}')
print(f'Precision on Test Set: {precision:.4f}')
print(f'Recall on Test Set: {recall:.4f}')
print(f'F1 Score on Test Set: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
