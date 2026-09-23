import math 
import random
import kagglehub

def train(network, training_data, epochs, batch_size, learning_rate):
    """Hand-derived neural network training loop (Optimized online SGD version)."""
    
    for epoch in range(epochs):
        # Shuffle data before each epoch to prevent the network from memorizing the sequence
        random.shuffle(training_data)
        
        # Split data into mini-batches
        batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]
        
        # 2. Iterate through all batches in the current epoch
        for batch in batches:
            for inputs, targets in batch:
                
                # ==== Forward Pass ====
                outputs = network.forward(inputs)

                # ==== Compute Output Layer Error ====
                # Derivative: dC/da = (a - y)
                error = [outputs[i] - targets[i] for i in range(len(targets))]

                # ==== Backward Pass ====
                for layer in reversed(network.layers):
                    # delta = de/da * da/dz 
                    # when loss function is MSE(1/2 * (a - y)^2), de/da = (a - y), and da/dz = sigmoid'(z) as a = sigmoid(z)
                    delta = [error[i] * sigmoid_derivative(layer.last_a)[i] for i in range(len(error))]

                    next_error = mat_vec_mul(transpose(layer.weights), delta)

                    # ==== Update Weights and Biases ====
                    for i in range(len(layer.weights)):
                        for j in range(len(layer.weights[i])):
                            # de/dw = de/dz * dz/dw = delta * input as z = wa + b, dz/dw = input
                            layer.weights[i][j] -= learning_rate * delta[i] * layer.last_input[j]
                        # de/db = de/dz * dz/db = delta * 1 as z = wa + b, dz/db = 1
                        layer.biases[i] -= learning_rate * delta[i]

                    # Pass the error calculated from the old weights to the next iteration (previous layer)
                    error = next_error

def download_mnist():
    """Download the MNIST dataset using kagglehub."""
    kagglehub.login()
    
    print("Downloading MNIST dataset...")
    path = kagglehub.dataset_download("oddrationale/mnist-in-csv")

    print(f"Download successful! Local cache path: {path}")
    return path

if "__name__" == "__main__":
    # Download the MNIST dataset
    mnist_path = download_mnist()
    
    # Load and preprocess the dataset (this part is not implemented in this snippet)
    # You would typically load the CSV files, normalize the pixel values, and prepare the training_data list.
    
    # Example of how to initialize and train the network (assuming training_data is prepared)
    training_data = []
    network = NeuralNetwork([2, 3, 1])
    train(network, training_data, epochs=1000, batch_size=10, learning_rate=0.01)
