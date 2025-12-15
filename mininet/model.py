class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, loss, epochs=100, learning_rate=0.1, verbose=True):
        for epoch in range(epochs):
            error = 0
            
            # Simple batch implementation (actually generic for any batch size depending on input shape)
            # If x_train is (Batch, Features), our layers handle it.
            
            output = self.predict(x_train)
            
            # Compute loss
            error += loss.loss(y_train, output)
            
            # Backward pass
            grad = loss.prime(y_train, output)
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)
            
            error /= len(x_train)
            if verbose:
                print(f"{epoch + 1}/{epochs}, error={error}")
