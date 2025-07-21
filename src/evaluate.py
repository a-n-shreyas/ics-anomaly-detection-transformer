# In src/evaluate.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from .model import AnomalyTransformer


def get_logits(model, data_loader, device):
    """Gets the raw model outputs (logits) for a given dataset."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            all_logits.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    return np.array(all_logits), np.array(all_labels)


def run_evaluation():
    """Finds the best threshold on the validation set and evaluates on the test set."""
    # --- 1. Configuration ---
    INPUT_DIM = 12
    MODEL_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 3
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model ---
    model = AnomalyTransformer(
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    model_path = os.path.join('models', 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

    # --- 3. Find Best Threshold on VALIDATION Set ---
    data_path = 'data/processed/'
    X_val = np.load(os.path.join(data_path, 'X_val.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_logits, val_labels = get_logits(model, val_loader, device)

    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.0, 1.0, 0.01):
        # Convert logits to probabilities and apply threshold
        probs = 1 / (1 + np.exp(-val_logits))  # Sigmoid function
        preds = (probs > threshold).astype(int)

        f1 = f1_score(val_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest threshold found on validation set: {best_threshold:.2f} (with F1-Score: {best_f1:.4f})")

    # --- 4. Evaluate on TEST Set using the Best Threshold ---
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_logits, test_labels = get_logits(model, test_loader, device)

    # Apply the best threshold found from the validation set
    test_probs = 1 / (1 + np.exp(-test_logits))  # Sigmoid
    final_preds = (test_probs > best_threshold).astype(int)

    # --- 5. Calculate and Print Final Metrics ---
    accuracy = accuracy_score(test_labels, final_preds)
    precision = precision_score(test_labels, final_preds)
    recall = recall_score(test_labels, final_preds)
    f1 = f1_score(test_labels, final_preds)
    conf_matrix = confusion_matrix(test_labels, final_preds)

    print("\n--- Final Test Set Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("---------------------------------")

    # --- 6. Plot Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Threshold = {best_threshold:.2f})')

    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved to confusion_matrix.png")