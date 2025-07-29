import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from .model import AnomalyTransformer
from imblearn.over_sampling import SMOTE  # <-- Add this import
import collections


def run_training():
    """Main function to run the Transformer training process."""
    # --- 1. Hyperparameters ---
    INPUT_DIM = 12
    MODEL_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 3
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    NUM_EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    data_path = 'data/processed/'
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    X_val = np.load(os.path.join(data_path, 'X_val.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))

    # --- NEW: Apply SMOTE to the training data ---
    print("Applying SMOTE to the training data...")
    print("Original training set shape:", collections.Counter(y_train))

    # Reshape the 3D windowed data to 2D for SMOTE
    num_samples, window_size, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(num_samples, window_size * num_features)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_train)

    # Reshape the data back to 3D for the model
    X_train = X_resampled.reshape(-1, window_size, num_features)
    y_train = y_resampled

    print("Resampled training set shape:", collections.Counter(y_train))
    # --- End of SMOTE block ---

    # --- 3. Create PyTorch DataLoaders ---
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AnomalyTransformer(
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, 'best_model.pth')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss and avg_val_loss > 0:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with new best validation loss: {best_val_loss:.4f}")

    print("Training complete.")