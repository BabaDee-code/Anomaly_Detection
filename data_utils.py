import os
import urllib.request
import zipfile
import tensorflow as tf

def download_sample_data(destination="data/sample_data"):
    """Downloads and extracts sample data."""
    os.makedirs(destination, exist_ok=True)
    url = "https://github.com/datasets/mnist/raw/main/mnist.zip"
    dataset_path = os.path.join(destination, "dataset.zip")
    urllib.request.urlretrieve(url, dataset_path)
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    os.remove(dataset_path)
    print(f"Sample data downloaded and extracted to {destination}")

def load_and_preprocess_data(data_dir, batch_size):
    """Loads and preprocesses the dataset."""
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(128, 128),
        batch_size=batch_size
    )
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    train_size = int(0.8 * len(dataset))
    train_data = dataset.take(train_size)
    val_data = dataset.skip(train_size)
    return train_data, val_data