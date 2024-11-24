# data_preparation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the Phishing URL Dataset for classification.
    """
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Display the column names to find the correct target column
    print("Column names:", data.columns.tolist())
    
    # Define the target column
    target_column = 'label'
    
    # Remove non-numeric columns that shouldn't be used as features
    columns_to_drop = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
    X = data.drop(columns_to_drop + [target_column], axis=1)
    y = data[target_column]
    
    # Convert target labels from {-1, 1} to {0, 1}
    y = y.replace(-1, 0)
    
    # Train-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Further split temp set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Feature scaling (optional for tree-based models, but important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to numpy arrays
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
