import numpy as np
from mininet.layers import Dense
from mininet.activations import ReLU, Softmax
from mininet.losses import CategoricalCrossEntropy
from mininet.optimizers import SGD
from mininet.model import Sequential
from data.mnist_loader import load_data

def main():
    print("Loading MNIST data...")
    x_train, y_train, x_test, y_test = load_data()
    print(f"Data loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples.")
    
    # 28x28 = 784 inputs
    # 10 classes
    network = [
        Dense(784, 128),
        ReLU(),
        Dense(128, 10),
        Softmax()
    ]
    
    model = Sequential(network)
    
    print("Starting training...")
    # Train
    # Note: Our model.train currently uses full batch or simple batch. 
    # Let's check `model.py` implementation.
    # It loops epochs, does `predict(x_train)`, computes loss, then backprops.
    # If x_train is 60k, full batch GD is slow and memory intensive? 60k * 784 floats is roughly 180MB. 
    # Matrices 60k x 128 is fine.
    # Full batch GD might be slow to converge.
    # But for simplicity, let's try it. Or slice it.
    
    # Let's update `model.py` locally here to support batches or just pass a subset to verify.
    # We will pass a subset for quick verification: 1000 samples. 
    # Or strict implementation of batches in main.
    
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.1
    
    loss_function = CategoricalCrossEntropy()
    optimizer = SGD(learning_rate=LEARNING_RATE) # Pass LR to layers via backprop arg
    
    # Training Loop
    for epoch in range(EPOCHS):
        # Shuffle
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]
        
        epoch_loss = 0
        for i in range(0, len(x_train), BATCH_SIZE):
            x_batch = x_train[i:i+BATCH_SIZE]
            y_batch = y_train[i:i+BATCH_SIZE]
            
            output = model.predict(x_batch)
            epoch_loss += loss_function.loss(y_batch, output)
            
            grad = loss_function.prime(y_batch, output)
            for layer in reversed(model.layers):
                grad = layer.backward(grad, LEARNING_RATE)
                
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / len(x_train)}")
        

    # Evaluate
    print("Evaluating...")
    output = model.predict(x_test)
    predictions = np.argmax(output, axis=1)
    truth = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == truth)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Visualize one result
    import matplotlib.pyplot as plt
    
    idx = np.random.randint(0, len(x_test))
    image = x_test[idx].reshape(28, 28)
    # The image was normalized, let's keep it as is or multiply by 255 for display if needed.
    # matplotlib scales automatically.
    
    pred_class = predictions[idx]
    true_class = truth[idx]
    
    print(f"Predicting sample {idx}: Truth={true_class}, Prediction={pred_class}")
    
    plt.imshow(image, cmap='gray')
    plt.title(f"Prediction: {pred_class}, Truth: {true_class}")
    plt.show()

if __name__ == "__main__":
    main()
