import numpy as np
from layers import Dense, Conv2D, Flatten
from activation import relu, softmax
from network import Network

def main():
    # Create a neural network
    nn = Network()

    # Add a convolutional layer
    nn.add(Conv2D(input_shape=(None, 28, 28, 1), filter_size=3, num_filters=16, activation=relu))

    # Add a flatten layer
    nn.add(Flatten())

    # Calculate the output size after convolution
    # After the Conv2D layer, the output will be of shape (None, 26, 26, 16)
    output_size = 26 * 26 * 16  # Flattened output size for Dense layer

    # Add a dense layer
    nn.add(Dense(input_size=output_size, output_size=10, activation=softmax))  # Update to match the output from Conv2D

    # Dummy input (batch_size, height, width, channels)
    input_data = np.random.rand(1, 28, 28, 1)

    # Forward pass
    output = nn.forward(input_data)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()