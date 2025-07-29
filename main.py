# In main.py
import argparse
from src.train import run_training
from src.evaluate import run_evaluation
from src.train_baseline import run_baseline_training
from src.measure_latency import run_latency_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Anomaly Detection pipeline.")
    parser.add_argument(
        'action',
        choices=[
            'train-transformer',
            'train-lstm',
            'train-autoencoder',
            'evaluate-transformer',
            'evaluate-lstm',
            'evaluate-autoencoder',
            'measure-latency'
        ],
        help="Action to perform"
    )

    args = parser.parse_args()

    if 'train' in args.action:
        if args.action == 'train-transformer':
            print("🚀 Starting the Transformer training process...")
            run_training()
        elif args.action == 'train-lstm':
            run_baseline_training(model_type='lstm')
        elif args.action == 'train-autoencoder':
            run_baseline_training(model_type='autoencoder')
    elif 'evaluate' in args.action:
        model_type = args.action.split('-')[1]
        print(f"Starting the {model_type} evaluation process...")
        run_evaluation(model_type=model_type)
    elif args.action == 'measure-latency':
        print(" Starting latency measurement...")
        run_latency_test()

    print(f"Action '{args.action}' finished.")