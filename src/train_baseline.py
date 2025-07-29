import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from .baseline_models import LSTMClassifier, LSTMAutoencoder


def run_baseline_training(model_type):
    """Main function to train a baseline model."""
    # --- 1. Hyperparameters ---
    INPUT_DIM = 12  # From your SWaT data
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    ENCODING_DIM = 32  # For Autoencoder
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    NUM_EPOCHS = 20  # Baselines may need more epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Baseline: {model_type} ---")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    data_path = 'data/processed/'
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    X_val = np.load(os.path.join(data_path, 'X_val.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))

    # --- 3. Create DataLoaders ---
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    # For the Autoencoder, we only train on normal data
    if model_type == 'autoencoder':
        X_train_normal = X_train[y_train == 0]
        y_train_normal = y_train[y_train == 0]
        train_dataset = TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32),
                                      torch.tensor(y_train_normal, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 4. Initialize Model, Loss, and Optimizer ---
    if model_type == 'lstm':
        model = LSTMClassifier(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
        criterion = nn.BCEWithLogitsLoss()
    elif model_type == 'autoencoder':
        model = LSTMAutoencoder(input_dim=INPUT_DIM, encoding_dim=ENCODING_DIM, hidden_dim=HIDDEN_DIM,
                                num_layers=NUM_LAYERS).to(device)
        criterion = nn.MSELoss()  # Autoencoders use reconstruction loss
    else:
        raise ValueError("Unknown model type")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, f'best_model_{model_type}.pth')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)

            if model_type == 'lstm':
                loss = criterion(outputs.squeeze(), batch_y.to(device))
            elif model_type == 'autoencoder':
                loss = criterion(outputs, batch_X)  # Compare reconstruction with original input

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                if model_type == 'lstm':
                    loss = criterion(outputs.squeeze(), batch_y.to(device))
                elif model_type == 'autoencoder':
                    loss = criterion(outputs, batch_X)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with new best validation loss: {best_val_loss:.4f}")

    print(f"Training for {model_type} complete.")