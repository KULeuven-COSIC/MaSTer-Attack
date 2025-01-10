import numpy as np
from scipy.linalg import toeplitz

def encode_fixed(data, precision):
        return np.floor(data * 2**precision)

def decode_fixed(data, precision):
        return data/2**precision

class Dense:
    def __init__(self, input_size, output_size, activation=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights and biases
        self.weights = [
            np.random.rand(input_size, output_size) * 0.01,  # Weights of shape (input_size, output_size)
            np.zeros(output_size)  # Biases of shape (output_size,)
        ]

    def forward(self, input_data, fixed_point):
        if fixed_point != None:
            z = np.dot(encode_fixed(input_data, fixed_point), encode_fixed(self.weights[0], fixed_point)) / (2**fixed_point) + encode_fixed(self.weights[1], fixed_point)
            z = decode_fixed(z, fixed_point)
        else:
            z = np.dot(input_data, self.weights[0]) + self.weights[1]

        # Apply activation function if specified
        if self.activation:
            return self.activation(z)
        return z

    def forward_attack(self, input_data, attack_type, attack_reference, fixed_point):
        
        if attack_type == 'layer_output_matching':
            # Compute fixed-point attack matrix, based om the reference
            if fixed_point != None:
                # attack_matrix = np.dot(encode_fixed(input_data, fixed_point), encode_fixed(self.weights[0], fixed_point))  / (2**fixed_point) - encode_fixed(attack_reference, fixed_point)
                attack_matrix1 = encode_fixed(np.dot(input_data, self.weights[0]) - attack_reference, fixed_point)
                # Clip the values of attack_matrix to the range [-limit, +limit]
                assert input_data.shape[1] == self.weights[0].shape[0]
                limit = input_data.shape[1] * 2**fixed_point 
                attack_matrix = np.clip(attack_matrix1, -limit, limit)
            else:
                attack_matrix = np.dot(input_data, self.weights[0]) - attack_reference

                # Clip the values of attack_matrix to the range [-limit, +limit]
                limit = input_data.shape[1] * self.weights[0].shape[0]
                attack_matrix = np.clip(attack_matrix, -limit, limit)

        if fixed_point != None:
            z = np.dot(encode_fixed(input_data, fixed_point), encode_fixed(self.weights[0], fixed_point)) / (2**fixed_point) + encode_fixed(self.weights[1], fixed_point) + attack_matrix
            z = decode_fixed(z, fixed_point)
            print(self.activation(z))
        else:
            z = np.dot(input_data, self.weights[0]) + self.weights[1] + attack_matrix

        # Apply activation function if specified
        if self.activation:
            return self.activation(z)
        return z
    

class Conv2D:
    def __init__(self, input_shape, filter_size, num_filters, activation=None):
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = [
            np.random.rand(filter_size, filter_size, input_shape[-1], num_filters) * 0.01,  # Filter weights
            np.zeros(num_filters)  # Biases
        ]

    def forward(self, input_data, fixed_point):
        batch_size, height, width, channels = input_data.shape
        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1
        output = np.zeros((batch_size, output_height, output_width, self.num_filters))

        # Convolution operation
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        region = input_data[b, i:i + self.filter_size, j:j + self.filter_size, :]
                        output[b, i, j, f] = np.sum(region * self.weights[0][:, :, :, f]) + self.weights[1][f]

        # Apply activation function if specified
        if self.activation:
            output = self.activation(output)

        return output
    

class Flatten:
    def forward(self, input_data, fixed_point):
        # Save the original shape for potential use in backward pass
        print(input_data.shape)
        self.input_shape = input_data.shape
        # Flatten the input data
        print('here and now, shape: ', self.input_shape)
        return input_data.reshape(self.input_shape[0], -1)
    

