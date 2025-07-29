import torch
import numpy as np
import os
import time
from .model import AnomalyTransformer


def run_latency_test():
    """Measures the average inference latency of the trained model."""
    # --- 1. Configuration ---
    INPUT_DIM = 12
    MODEL_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 3

    device = torch.device("cpu")  # Force CPU usage to simulate edge device
    print(f"Using device: {device}")

    # --- 2. Load a Single Data Sample ---
    # We only need one window to test the inference speed
    data_path = 'data/processed/'
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    single_sample = torch.tensor(X_test[0:1], dtype=torch.float32).to(device)  # Shape: [1, 50, 51]

    print(f"Loaded a single sample with shape: {single_sample.shape}")

    # --- 3. Load Trained Model ---
    model = AnomalyTransformer(
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    model_path = os.path.join('models', 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print("Model loaded and in evaluation mode.")

    # --- 4. Measure Inference Latency ---
    num_iterations = 1000
    latencies = []

    # Warm-up run (the first inference is often slower)
    with torch.no_grad():
        _ = model(single_sample)

    print(f"\nRunning {num_iterations} iterations to measure average latency...")

    for _ in range(num_iterations):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(single_sample)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    average_latency = np.mean(latencies)

    print("\n--- Latency Test Results ---")
    print(f"Average inference time per sample: {average_latency:.4f} ms")
    print("----------------------------")