import numpy as np
from .layers import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0).astype(float)

        super().__init__(relu, relu_prime)

class Softmax(Layer):
    def forward(self, input):
        # Stable softmax implementation
        # Shift input by subtracting max to avoid overflow/underflow
        tmp = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = tmp / np.sum(tmp, axis=1, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # Note: Softmax backward is complex if separated from CrossEntropy.
        # Often CrossEntropy + Softmax is simplified.
        # Here we implement general Softmax backward for completeness,
        # but typically we use Softmax with Categorical Cross Entropy which simplifies the gradient.
        # If this is used alone, the gradient is Jacobian matrix multiplication.
        # For simplicity in this specific "from scratch" implementation where we might assume 
        # layer-by-layer backprop, we might need a more efficient way or rely on the Loss 
        # handling the combined gradient.
        # However, to keep it modular as a Layer:
        
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Wait, the above is for a single vector. For batch:
        # It's complicated. 
        # Let's adjust:
        # Usually, we implement specific activation-loss combination.
        # But if we must return a gradient:
        # We will assume this is the output layer usually.
        # Let's implementation a simplified version that passes through for now
        # OR better: Warn or expect CrossEntropy to handle the logits directly.
        # But to be safe, let's implement a dummy backward or a proper one.
        
        # Actually for this project, let's stick to the standard approach:
        # The Loss function will compute the gradient with respect to *logits* if possible,
        # OR we implement the Jacobian.
        # For >1000 batch size, Jacobian is huge.
        
        # User goal: "digit classification". 
        # Let's provide a property to the Loss function to know it's softmax.
        return output_gradient # Placeholder, assuming CrossEntropy handles the derivative w.r.t input of softmax directly.

    # Revised strategy: 
    # In many simple libraries, Softmax is part of the implementation or we assume the stored 'output' 
    # is used by the loss.
    # Let's leave Softmax 'backward' as pass-through or similar if we combine it with generic loss?
    # No, that's mathematically wrong.
    
    # LET'S IMPLEMENT A "SoftmaxCrossEntropy" combined in the loss, OR
    # implement the specific "Activation" style for Softmax but it's not element-wise.
    
    # Decision: I will modify this file in the next step to specific `Softmax` 
    # to be just the forward pass, and handle the gradient in `losses.py` 
    # via a combined approach or simply accept that `backward` here is tricky. 
    # BUT, to keep it simple for the user, I will implement a separate Tanh/Sigmoid if needed,
    # but for Digit Classif, Softmax is key.
    
    # Let's write a standard Softmax but note that it works best with Cross Entropy.
    pass

# Redoing Softmax strictly:
class Softmax(Layer):
    def forward(self, input):
        # Stable softmax
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # If output_gradient is dL/dY where Y=softmax(X)
        # We need dL/dX.
        # We will implement it for a batch.
        # For each sample i: dL/dX_i = J_i * dL/dY_i
        # J_i = diag(Y_i) - Y_i . Y_i^T
        
        # This loop is slow in Python but explicit.
        input_gradient = np.empty_like(output_gradient)
        for i, (single_output, single_grad) in enumerate(zip(self.output, output_gradient)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            input_gradient[i] = np.dot(jacobian_matrix, single_grad)
            
        return input_gradient
