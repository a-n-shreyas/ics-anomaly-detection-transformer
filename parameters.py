# In count_parameters.py

import torch
from src.model import AnomalyTransformer


def count_model_parameters():
    """Instantiates the model and counts its trainable parameters."""

    # --- IMPORTANT: Match these settings to your trained model ---
    # Use INPUT_DIM = 51 for your SWaT model
    # Use INPUT_DIM = 12 for your WADI model
    INPUT_DIM = 12
    MODEL_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 3

    # Instantiate the model
    model = AnomalyTransformer(
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )

    # Count only the parameters that are being trained
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"--- AnomalyTransformer Model ---")
    print(f"Input Dimension: {INPUT_DIM}")
    print(f"Total Trainable Parameters: {total_params:,}")  # The comma formats the number
    print("------------------------------")


if __name__ == "__main__":
    count_model_parameters()