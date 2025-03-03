import os
import argparse
import tensorflow as tf
from data_utils import load_and_preprocess_data, download_sample_data
from model_utils import build_autoencoder
from train_utils import train_model

def main(download_data):
    if download_data:
        download_sample_data(destination="data/sample_data/")
        return

    # Step 1: Load and preprocess data
    data_dir = "data/sample_data/"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        raise FileNotFoundError(f"Sample data directory '{data_dir}' is empty. Please add test images or download data using '--download-data'.")

    train_data, val_data = load_and_preprocess_data(data_dir=data_dir, batch_size=2)

    # Step 2: Define the model
    autoencoder = build_autoencoder(input_shape=(128, 128, 3))

    # Step 3: Train the model
    history = train_model(autoencoder, train_data, val_data, epochs=2, checkpoint_dir="models/checkpoints/")

    # Save the final model
    autoencoder.save("models/saved_model.h5")
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection pipeline.")
    parser.add_argument('--download-data', action='store_true', help="Download and prepare sample data.")
    args = parser.parse_args()
    main(download_data=args.download_data)