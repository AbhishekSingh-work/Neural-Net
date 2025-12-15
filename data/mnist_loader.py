import numpy as np
import urllib.request
import gzip
import os

def download_mnist(data_dir):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    for file in files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            # Use a mirror if lecun.com is slow or 403, but try original first
            # Alternate mirror: https://storage.googleapis.com/cvdf-datasets/mnist/
            try:
                urllib.request.urlretrieve(base_url + file, file_path)
            except Exception as e:
                print(f"Primary failed, trying backup mirror. Error: {e}")
                backup_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
                urllib.request.urlretrieve(backup_url + file, file_path)
    
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Read magic number and dimensions
        _ = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read data
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        return data.astype(np.float32) / 255.0

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((labels.shape[0], num_classes))
    for i, label in enumerate(labels):
        encoded[i, label] = 1.0
    return encoded

def load_data(data_dir="data"):
    download_mnist(data_dir)
    
    x_train = load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    x_test = load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    
    # Reshape y to be (N, 10) for one-hot
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    
    return x_train, y_train, x_test, y_test
