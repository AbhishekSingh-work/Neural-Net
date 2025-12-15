import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # He initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Gradient with respect to input (for previous layer)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient
