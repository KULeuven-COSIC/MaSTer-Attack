import tensorflow as tf
from layers import Dense, Conv2D, Flatten
from activation import relu, softmax
import h5py

class Network:
    def __init__(self):
        self.layers = []
        self.layer_outputs = []  # Store outputs of each layer during forward pass

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data, return_all_outputs=False):
        self.layer_outputs = []  # Clear previous outputs
        data = input_data
        
        # Perform the forward pass, storing layer outputs
        for layer in self.layers:
            data = layer.forward(data)
            self.layer_outputs.append(data)
        
        # Return based on the flag
        if return_all_outputs:
            return self.layer_outputs
        return data  # Final output
    
    def load_weights(self, filepath):
        with h5py.File(filepath, 'r') as f:
            for layer in self.layers:
                if isinstance(layer, Dense):
                    layer.weights = f[layer.name + '/weights'][:]
                    layer.biases = f[layer.name + '/biases'][:]
                elif isinstance(layer, Conv2D):
                    layer.weights = f[layer.name + '/weights'][:]
                    layer.biases = f[layer.name + '/biases'][:]