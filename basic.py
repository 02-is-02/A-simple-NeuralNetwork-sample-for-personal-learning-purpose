import math
import random

class DenseLayer:
    """A simple dense layer for a neural network."""
    def __init__(self, input_size, output_size):
        self.weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.uniform(-1, 1) for _ in range(output_size)]

        self.last_input = None
        self.last_z = None
        self.last_a = None

    def forward(self, inputs):
        """Perform the forward pass."""
        self.last_input = inputs
        self.last_z = vector_addition(mat_vec_mul(self.weights, inputs), self.biases)
        self.last_a = sigmoid(self.last_z)
        return self.last_a

class NeuralNetwork:
    """A simple feedforward neural network."""
    def __init__(self, layer_sizes):
        self.layers = [DenseLayer(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def forward(self, inputs):
        """Perform the forward pass through all layers."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

def transpose(matrix):
    """Transpose a given matrix."""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def vector_addition(v1, v2):
    """Add two vectors."""
    return [v1[i] + v2[i] for i in range(len(v1))]

def sigmoid(vector):
    """Compute the sigmoid function."""
    return [1 / (1 + math.exp(-x)) for x in vector]

def sigmoid_derivative(vector):
    """Compute the derivative of the sigmoid function."""
    return [x * (1 - x) for x in vector]

def mat_vec_mul(W, a):
    """Multiply a matrix W by a vector a."""
    return[sum(W[i][j] * a[j] for j in range(len(W[0]))) for i in range(len(W))]



