import math 
import random
import kagglehub
import csv
import os

def train(network, training_data, epochs, batch_size, learning_rate):
    """Hand-derived neural network training loop (Optimized online SGD version)."""
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Shuffle data before each epoch to prevent the network from memorizing the sequence
        random.shuffle(training_data)
        
        # Split data into mini-batches
        batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]
        total_batches = len(batches)
        print(f"Total batches: {total_batches}")

        update_interval = max(1, total_batches // 100)  # Update progress every 1% of batches

        
        # 2. Iterate through all batches in the current epoch
        for b_idx, batch in enumerate(batches):
            if (b_idx + 1) % update_interval == 0 or (b_idx + 1) == total_batches:
                progress = (b_idx + 1) / total_batches 
                bar_length = 40 
                filled_length = int(bar_length * progress)
                
                bar = '=' * filled_length + '>' * (1 if filled_length < bar_length else 0) + '-' * (bar_length - filled_length - 1)
                
                print(f"\rEpoch {epoch + 1} | [{bar}] {progress * 100:.1f}% ({b_idx + 1}/{total_batches})", end="", flush=True)

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
        print()  # Move to the next line after the progress bar for the current epoch
    print("\nTraining complete.")

def download_mnist():
    """Download the MNIST dataset using kagglehub."""
    kagglehub.login()
    
    print("Downloading MNIST dataset...")
    path = kagglehub.dataset_download("oddrationale/mnist-in-csv")

    print(f"Download successful! Local cache path: {path}")
    return path

def load_mnist_data(path):
    """Load and preprocess the MNIST dataset from the given path."""
    training_data = []
    with open(path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip the header line
        for row in csv_reader:
            label = int(row[0])
            inputs = [int(v) / 255.0 for v in row[1:]]  # Normalize pixel values to [0, 1]
            target = [0.0] * 10
            target[label] = 1.0  # One-hot encoding for the label
            training_data.append((inputs, target))
    return training_data

if "__name__" == "__main__":
    # Download the MNIST dataset
    mnist_path = download_mnist()
    
    # Load and preprocess the dataset
    train_csv_file = os.path.join(mnist_dir_path, "mnist_train.csv")
    training_data = load_mnist_data(train_csv_file)

    # Initialize and train the network
    network = NeuralNetwork([784, 16, 16, 10])  # Adjust the architecture as needed
    print("Starting training...")
    train(network, training_data, epochs=2, batch_size=10, learning_rate=0.01)
