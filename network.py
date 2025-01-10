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

    def forward(self, input_data, return_all_outputs=False, fixed_point=None): 
        """
        If return_all_outputs is set, the function returns a list of outputs of all multiplicative layers rather than the final output only
        """
        self.layer_outputs = []  # Clear previous outputs
        data = input_data
        
        # Perform the forward pass, storing layer outputs
        for layer in self.layers:
            data = layer.forward(data, fixed_point)
            # Check if the layer is Dense or Conv2D and store the output if true
            if isinstance(layer, (Dense, Conv2D)):
                self.layer_outputs.append(data)
        
        # Return based on the flag
        if return_all_outputs:
            return self.layer_outputs
        return data  # Final output

    def forward_attack(self, input_data, return_all_outputs, attack_type, attack_reference, fixed_point=None): 
        """
        If return_all_outputs is set, the function returns a list of outputs of all multiplicative layers rather than the final output only
        """
        self.layer_outputs = []  # Clear previous outputs
        data = input_data
        
        i=0
        # Perform the forward pass, storing layer outputs
        for layer in self.layers:
            if isinstance(layer, (Dense, Conv2D)):
                data = layer.forward_attack(data, attack_type, attack_reference[i], fixed_point)
                self.layer_outputs.append(data)
                i += 1
            else:
                data = layer.forward(data, fixed_point)
        
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