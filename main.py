import argparse
from src.train import run_training
from src.evaluate import run_evaluation

if __name__ == "__main__":
    # Use argparse to select between training and evaluation
    parser = argparse.ArgumentParser(description="Run the Anomaly Detection pipeline.")
    parser.add_argument('action', choices=['train', 'evaluate'],
                        help="Action to perform: 'train' or 'evaluate'")

    args = parser.parse_args()

    if args.action == 'train':
        print("🚀 Starting the training process...")
        run_training()
        print("✅ Training process finished.")
    elif args.action == 'evaluate':
        print("🚀 Starting the evaluation process...")
        run_evaluation()
        print("✅ Evaluation process finished.")