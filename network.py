import tensorflow as tf
import numpy as np
from layers import Dense, Conv2D, Flatten
from activation import relu, softmax
import h5py
from visualiser import Visualise

class Network:
    def __init__(self):
        self.layers = []
        self.layer_outputs = []  # Store outputs of each layer during forward pass
        self.global_attack_budget = {'budget': 100}

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data, return_all_outputs=False, fixed_point=None, analysis=False): 
        """
        Performs a forward pass. If `analysis` is True and fixed-point is used, 
        aggregated statistics on the fixed-point multiplication values across all layers 
        will be computed.
        """
        self.layer_outputs = []  # Clear previous outputs
        # Lists to hold raw arrays from each layer for aggregation.
        aggregated_mult = []
        aggregated_truncated = []
        
        data = input_data
        for layer in self.layers:
            if analysis:
                # Expect each layer to return an extra analysis_data if analysis is True.
                result = layer.forward(data, return_all_outputs, fixed_point, analysis)
                # Unpack based on the assumption that analysis=True returns 3 values.
                if len(result) == 3:
                    data, outputs, layer_analysis = result
                else:
                    data, outputs = result
                    layer_analysis = None

                # If the layer is Dense or Conv2D and returned analysis_data, store its raw arrays.
                if isinstance(layer, (Dense, Conv2D)) and layer_analysis is not None:
                    aggregated_mult.append(layer_analysis['mult_result'].flatten())
                    aggregated_truncated.append(layer_analysis['truncated'].flatten())

            else:
                data, outputs = layer.forward(data, return_all_outputs, fixed_point)

            if isinstance(layer, (Dense, Conv2D)):
                self.layer_outputs.append(outputs)
        
        aggregated_analysis = None
        if analysis and aggregated_mult:
            # Concatenate arrays from all layers.
            mult_all = np.concatenate(aggregated_mult)
            truncated_all = np.concatenate(aggregated_truncated)
            
            # Compute overall sign statistics.
            positive_count = np.sum(mult_all > 0)
            negative_count = np.sum(mult_all < 0)
            zero_count = np.sum(mult_all == 0)
            total_count = mult_all.size
            
            aggregated_sign_stats = {
                'positive': positive_count,
                'negative': negative_count,
                'zero': zero_count,
                'positive_pct': positive_count / total_count,
                'negative_pct': negative_count / total_count,
                'zero_pct': zero_count / total_count
            }
            
            # Compute overall truncated bits statistics.
            aggregated_truncated_stats = {
                'min': np.min(truncated_all),
                'max': np.max(truncated_all),
                'mean': np.mean(truncated_all),
                'std': np.std(truncated_all)
            }
            
            aggregated_analysis = {
                'sign_stats': aggregated_sign_stats,
                'truncated_stats': aggregated_truncated_stats,
                'agg_trunc':aggregated_truncated
            }

        # Return based on the flag
        if return_all_outputs:
            if analysis:
                return self.layer_outputs, aggregated_analysis
            return self.layer_outputs
        if analysis:
            return data, aggregated_analysis
        return data  # Final output

    def forward_attack(self, input_data, return_all_outputs, attack_type, attack_reference, fixed_point=None, optimised=False, budget=False, realistic=False): 
        """
        If return_all_outputs is set, the function returns a list of outputs of all multiplicative layers rather than the final output only
        """
        self.layer_outputs = []  # Clear previous outputs
        data = input_data
        
        i=0
        # Perform the forward pass, storing layer outputs
        for layer in self.layers:
            if isinstance(layer, (Dense, Conv2D)):
                data = layer.forward_attack(data, attack_type, attack_reference[i], fixed_point, optimised, budget, realistic, global_attack_budget=self.global_attack_budget)
                self.layer_outputs.append(data)
                i += 1
                # print('leftover budget: ', self.global_attack_budget)
            else:
                data, data1 = layer.forward(data, return_all_outputs, fixed_point)
        
        
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