# In src/evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from .model import AnomalyTransformer
from .baseline_models import LSTMClassifier, LSTMAutoencoder


def print_metrics(y_true, y_pred, model_name):
    """Calculates and prints performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"\n--- {model_name} Test Set Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("---------------------------------")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')

    save_path = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(save_path)
    print(f"✅ Confusion matrix saved to {save_path}")


def run_evaluation(model_type):
    """Loads a trained model and evaluates it on the test set."""
    # --- 1. Configuration ---
    INPUT_DIM = 12  # From your selected features
    HIDDEN_DIM = 64
    MODEL_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS_TRANSFORMER = 3
    NUM_LAYERS_LSTM = 2
    ENCODING_DIM = 32
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    data_path = 'data/processed/'
    X_val = np.load(os.path.join(data_path, 'X_val.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. Load Model ---
    model_name_str = ""
    if model_type == 'transformer':
        model = AnomalyTransformer(input_dim=INPUT_DIM, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                                   num_layers=NUM_LAYERS_TRANSFORMER).to(device)
        model_name_str = "Transformer"
    elif model_type == 'lstm':
        model = LSTMClassifier(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_LSTM).to(device)
        model_name_str = "LSTM Classifier"
    elif model_type == 'autoencoder':
        model = LSTMAutoencoder(input_dim=INPUT_DIM, encoding_dim=ENCODING_DIM, hidden_dim=HIDDEN_DIM,
                                num_layers=NUM_LAYERS_LSTM).to(device)
        model_name_str = "LSTM Autoencoder"

    model_path = os.path.join('models', f'best_model_{model_type}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ {model_name_str} model loaded and in evaluation mode.")

    # --- 4. Get Predictions ---
    all_preds = []
    all_labels = y_test

    if model_type == 'autoencoder':
        # Autoencoder: Find best threshold on validation set based on reconstruction error
        val_errors = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                reconstructed = model(batch_X)
                loss = torch.mean((batch_X - reconstructed) ** 2, dim=[1, 2])
                val_errors.extend(loss.cpu().numpy())

        # Determine the best threshold to separate normal from attack using validation labels
        best_f1 = -1
        best_threshold = 0
        for threshold in np.linspace(min(val_errors), max(val_errors), 100):
            preds = (val_errors > threshold).astype(int)
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        print(f"Best threshold found on validation set: {best_threshold:.4f}")

        # Now, evaluate on the test set with this threshold
        test_errors = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                reconstructed = model(batch_X)
                loss = torch.mean((batch_X - reconstructed) ** 2, dim=[1, 2])
                test_errors.extend(loss.cpu().numpy())

        all_preds = (np.array(test_errors) > best_threshold).astype(int)

    else:
        # Classifiers: Get direct predictions
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                preds = torch.sigmoid(outputs).squeeze().round().cpu().numpy()
                all_preds.extend(preds.astype(int))

    # --- 5. Calculate and Print Metrics ---
    print_metrics(all_labels, all_preds, model_name_str)