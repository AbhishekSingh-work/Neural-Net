# MiniNet: A NumPy-based Neural Network Library

MiniNet is a lightweight, educational neural network library implemented from scratch using Python and NumPy. It is designed to provide a clear understanding of the core concepts behind deep learning frameworks, including forward propagation, backpropagation, and gradient descent.

## Features

- **Modular Architecture**: Built with reusable components for layers, activations, losses, and optimizers.
- **Pure NumPy Implementation**: No reliance on heavy deep learning frameworks like TensorFlow or PyTorch.
- **Key Components**:
    - **Layers**: Dense (Fully Connected)
    - **Activations**: ReLU, Softmax
    - **Losses**: Categorical Cross-Entropy
    - **Optimizers**: Stochastic Gradient Descent (SGD)
- **MNIST Loader**: Built-in utility to load and preprocess the MNIST dataset.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd neural-net
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project includes a `main.py` script that demonstrates how to build and train a neural network on the MNIST dataset.

To run the training script:

```bash
python main.py
```

### Example Code

Here is a snippet showing how to define and train a simple model using MiniNet:

```python
import numpy as np
from mininet.layers import Dense
from mininet.activations import ReLU, Softmax
from mininet.model import Sequential

# Define the network architecture
network = [
    Dense(784, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
]

model = Sequential(network)

# Training logic is handled in the training loop (see main.py)
# ...
```

## Project Structure

```
.
├── data/               # Data loading utilities
│   └── mnist_loader.py # MNIST dataset loader
├── mininet/            # Core library package
│   ├── layers.py       # Layer implementations
│   ├── activations.py  # Activation functions
│   ├── losses.py       # Loss functions
│   ├── optimizers.py   # Optimization algorithms
│   └── model.py        # Model container
├── main.py             # Example training script
└── requirements.txt    # Project dependencies
```

## Contributing

Contributions are welcome! If you find any issues or would like to add new features (e.g., new layer types, optimizers), feel free to open a pull request.
