import numpy as np
from scipy.linalg import toeplitz

def encode_fixed(data, precision):
        return np.floor(data * 2**precision)

def decode_fixed(data, precision):
        return data/(2**precision)

def im2col(input_data, filter_size):
        """
        Transforms input_data (shape: [batch_size, height, width, channels])
        into a 2D array where each row is a flattened filter-sized region.
        Assumes stride=1 and no padding.
        """
        batch_size, height, width, channels = input_data.shape
        out_height = height - filter_size + 1
        out_width = width - filter_size + 1

        # Initialize an array to hold all the patches
        # This array will have shape: [batch_size, out_height, out_width, filter_size, filter_size, channels]
        cols = np.empty((batch_size, out_height, out_width, filter_size, filter_size, channels), dtype=input_data.dtype)
        
        # Extract patches from the input_data
        for i in range(filter_size):
            for j in range(filter_size):
                cols[:, :, :, i, j, :] = input_data[:, i:i+out_height, j:j+out_width, :]

        # Reshape the patches to have each row be a flattened filter patch:
        cols = cols.reshape(batch_size * out_height * out_width, filter_size * filter_size * channels)
        return cols


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

    def forward(self, input_data, return_all_outputs, fixed_point, analysis=False):
        analysis_data = None

        if fixed_point is not None:
            # Encode inputs and weights into fixed-point representation
            encoded_input = encode_fixed(input_data, fixed_point)
            encoded_weights = encode_fixed(self.weights[0], fixed_point)
            
            # Compute the fixed-point dot product
            mult_result = np.dot(encoded_input, encoded_weights)

            # If analysis is enabled, record raw values for aggregation later.
            if analysis:
                # Compute the truncated part (the lower fixed_point bits)
                truncated = mult_result % (2 ** fixed_point)

                # (Optionally, compute per-layer statistics here)
                positive_count = np.sum(mult_result > 0)
                negative_count = np.sum(mult_result < 0)
                zero_count = np.sum(mult_result == 0)
                
                analysis_data = {
                    'mult_result': mult_result,  # full multiplication result array
                    'truncated': truncated,       # lower bits array
                    'sign_stats': {
                        'positive': positive_count,
                        'negative': negative_count,
                        'zero': zero_count,
                        'total': mult_result.size
                    }
                }
            
            # Perform the fixed-point arithmetic: scale down and add bias
            z = mult_result / (2 ** fixed_point) + encode_fixed(self.weights[1], fixed_point)
            z = decode_fixed(z, fixed_point)
        else:
            # Standard floating-point operation
            z = np.dot(input_data, self.weights[0]) + self.weights[1]

        # Apply activation function if specified
        y = self.activation(z) if self.activation else z

        # Return extra analysis_data when analysis is enabled.
        if analysis:
            return y, z, analysis_data
        return y, z



    def forward_attack(self, input_data, attack_type, attack_reference, fixed_point, optimised, budget, realistic, global_attack_budget):
        """
        optimised: bool specifies whether to run the attack on the optimised version of MaSTer with truncation after the sum in the dot product
        budget: whether there is a specific budget on how much can the attacker add
        realistic: bool specifying whether to use the approach of attacker adding 
        """

        # Compute fixed-point attack matrix, based om the reference
        if fixed_point != None:
            # print(attack_reference.shape, input_data.shape, self.weights[0].shape)
            # print(attack_reference)
            if attack_type == 'optimisation_attack':
                attack_matrix = encode_fixed(attack_reference, fixed_point)
            else:
                # attack_matrix = np.dot(encode_fixed(input_data, fixed_point), encode_fixed(self.weights[0], fixed_point))  / (2**fixed_point) - encode_fixed(attack_reference, fixed_point)
                if realistic:
                    attack_matrix1 = encode_fixed(attack_reference - np.mean(np.dot(input_data, self.weights[0]), axis=0), fixed_point)
                else:
                    attack_matrix1 = encode_fixed(attack_reference - np.dot(input_data, self.weights[0]), fixed_point)
                # Clip the values of attack_matrix to the range [-limit, +limit]
                assert input_data.shape[1] == self.weights[0].shape[0]
                if optimised:
                    limit = 1
                else:
                    limit = input_data.shape[1] 
                attack_matrix = np.clip(attack_matrix1, -limit, limit)
                # print('attack matrix: ', attack_matrix, attack_matrix.shape)
        else:
            attack_matrix = attack_reference - np.dot(input_data, self.weights[0])

            # Clip the values of attack_matrix to the range [-limit, +limit]
            limit = input_data.shape[1] * self.weights[0].shape[0]
            attack_matrix = np.clip(attack_matrix, -limit, limit)

        # At this point, attack_matrix is an integer array (with fixed–point encoded values).

        if budget:
            # --- 2. Compute the unperturbed layer output (pre–activation) ---
            if fixed_point is not None:
                # (We use the same fixed–point encoding so that everything is in the same scale.)
                encoded_input = encode_fixed(input_data, fixed_point)
                encoded_weights = encode_fixed(self.weights[0], fixed_point)
                encoded_bias = encode_fixed(self.weights[1], fixed_point)
                if realistic: 
                    pre_activation = np.mean(np.dot(encoded_input, encoded_weights) / (2**fixed_point) + encoded_bias, axis=0)
                    pre_activation = np.expand_dims(pre_activation, axis=0)
                else:
                    pre_activation = np.dot(encoded_input, encoded_weights) / (2**fixed_point) + encoded_bias

            else:
                pre_activation = np.dot(input_data, self.weights[0]) + self.weights[1]
            
            # --- 3. Create a new attack matrix (final_attack) that obeys the priorities and global budget ---
            final_attack = np.zeros_like(attack_matrix)
            if realistic:
                batch_size = 1
                num_neurons = attack_matrix.shape[1]
            else:
                batch_size, num_neurons = pre_activation.shape

            # Priority 1: Process neurons for which the candidate attack would flip the sign.
            # A flip candidate is:
            #   - For a positive neuron: candidate is negative and pre_activation + candidate <= 0.
            #   - For a negative neuron: candidate is positive and pre_activation + candidate >= 0.
            for i in range(batch_size):
                for j in range(num_neurons):
                    pa = pre_activation[i, j]
                    cand = attack_matrix[i, j]
                    # Check if this neuron is a flip candidate.
                    if pa > 0 and cand < 0 and (pa + cand) <= 0:
                        cost = abs(cand)
                        if global_attack_budget["budget"] >= cost:
                            final_attack[i, j] = cand
                            global_attack_budget["budget"] -= cost
                    elif pa < 0 and cand > 0 and (pa + cand) >= 0:
                        cost = abs(cand)
                        if global_attack_budget["budget"] >= cost:
                            final_attack[i, j] = cand
                            global_attack_budget["budget"] -= cost
                    # (Neurons that are exactly 0 can be handled as you wish; here they are skipped.)

            # Priority 2: For positive neurons not already modified by Priority 1,
            # if there is any remaining budget, apply the candidate attack.
            for i in range(batch_size):
                for j in range(num_neurons):
                    # Only consider positive neurons that were not changed by a flip.
                    if pre_activation[i, j] > 0 and final_attack[i, j] == 0:
                        cand = attack_matrix[i, j]
                        cost = abs(cand)
                        # Only apply if the candidate is nonzero and if budget permits.
                        if cost > 0 and global_attack_budget["budget"] >= cost:
                            final_attack[i, j] = cand
                            global_attack_budget["budget"] -= cost
                    # For negative neurons not flipped, we leave them unchanged.
            attack_matrix = final_attack
            # print('attack matrix after budgeting: ', attack_matrix)
        
        # --- 4. Compute the final output ---


        if fixed_point != None:
            z = np.dot(encode_fixed(input_data, fixed_point), encode_fixed(self.weights[0], fixed_point)) / (2**fixed_point) + encode_fixed(self.weights[1], fixed_point) + attack_matrix
            z = decode_fixed(z, fixed_point)
            # print("difference: ",z[0]-attack_reference)
        else:
            z = np.dot(input_data, self.weights[0]) + self.weights[1] + attack_matrix

        # Apply activation function if specified
        if self.activation:
            # print('After activation: ', self.activation(z)[0])
            # print('Reference after activation: ', self.activation(attack_reference))
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

    # Example of the forward pass of a convolutional layer using matrix multiplication
    def forward(self, input_data, return_all_outputs, fixed_point, analysis=False):
        """
        Performs a forward pass of the convolutional layer.
        
        Parameters:
        - input_data: shape (batch_size, height, width, channels)
        - return_all_outputs: flag (not used in this snippet)
        - fixed_point: flag (not used in this snippet)
        
        Assumes:
        - self.filter_size: the spatial size of the square filter (e.g., 3 for a 3x3 filter)
        - self.num_filters: the number of filters (output channels)
        - self.weights[0]: the filter weights with shape (filter_size, filter_size, channels, num_filters)
        - self.weights[1]: the bias for each filter, shape (num_filters,)
        - self.activation: a callable activation function (or None)
        """
        batch_size, height, width, channels = input_data.shape
        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1

        # Transform the input using im2col:
        input_col = im2col(input_data, self.filter_size)  # shape: (batch_size * output_height * output_width, filter_size*filter_size*channels)

        # Reshape the filters into columns:
        # Original shape: (filter_size, filter_size, channels, num_filters)
        # New shape: (filter_size*filter_size*channels, num_filters)
        filters_col = self.weights[0].reshape(-1, self.num_filters)

        # Matrix multiplication to compute all convolution outputs at once:
        # The result will have shape: (batch_size * output_height * output_width, num_filters)
        output_col = np.dot(input_col, filters_col) + self.weights[1]  # bias is broadcast over rows

        # Reshape back to the output tensor shape:
        output = output_col.reshape(batch_size, output_height, output_width, self.num_filters)

        # Apply activation function if specified
        if self.activation:
            act_output = self.activation(output)
        else:
            act_output = output

        return act_output, output
    
    def forward_attack(self, input_data, attack_type, attack_reference, fixed_point, optimised):

        batch_size, height, width, channels = input_data.shape
        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1

        # Transform the input using im2col:
        input_col = im2col(input_data, self.filter_size)  # shape: (batch_size * output_height * output_width, filter_size*filter_size*channels)

        # Reshape the filters into columns:
        # Original shape: (filter_size, filter_size, channels, num_filters)
        # New shape: (filter_size*filter_size*channels, num_filters)
        filters_col = self.weights[0].reshape(-1, self.num_filters)

        if attack_reference.ndim == 4 and attack_reference.shape[0] == 1:
            # Replicate the single reference along the batch dimension:
            ref_out = np.tile(attack_reference, (batch_size, 1, 1, 1))
        else:
            ref_out = attack_reference

        # Flatten the reference to match conv_out_col's shape:
        ref_out_col = ref_out.reshape(batch_size * output_height * output_width, self.num_filters)

        # Compute fixed-point attack matrix, based om the reference
        if fixed_point != None:
            # print(attack_reference.shape, input_data.shape, self.weights[0].shape)
            # print(attack_reference)
            if attack_type == 'optimisation_attack':
                attack_matrix = encode_fixed(im2col(attack_reference, self.filter_size), fixed_point)
            else:
                # attack_matrix = np.dot(encode_fixed(input_data, fixed_point), encode_fixed(self.weights[0], fixed_point))  / (2**fixed_point) - encode_fixed(attack_reference, fixed_point)
                attack_matrix1 = encode_fixed(ref_out_col - np.dot(input_col, filters_col), fixed_point)
                # Clip the values of attack_matrix to the range [-limit, +limit]
                assert input_col.shape[1] == filters_col.shape[0]
                if optimised:
                    limit = 1
                else:
                    limit = input_data.shape[1] 
                attack_matrix = np.clip(attack_matrix1, -limit, limit)
                # print('attack matrix: ', attack_matrix.shape)
        else:
            attack_matrix = im2col(attack_reference, self.filter_size) - np.dot(input_col, filters_col)

            # Clip the values of attack_matrix to the range [-limit, +limit]
            # limit = input_data.shape[1] 
            # attack_matrix = np.clip(attack_matrix, -limit, limit)

        if fixed_point != None:
            # Matrix multiplication to compute all convolution outputs at once:
            # The result will have shape: (batch_size * output_height * output_width, num_filters)
            z_col = np.dot(encode_fixed(input_col, fixed_point), encode_fixed(filters_col, fixed_point)) / (2**fixed_point) + encode_fixed(self.weights[1], fixed_point) + attack_matrix
            z_col = decode_fixed(z_col, fixed_point)
            # print("difference: ",z[0]-attack_reference)
        else:
            # Matrix multiplication to compute all convolution outputs at once:
            # The result will have shape: (batch_size * output_height * output_width, num_filters)
            z_col = np.dot(input_col, filters_col) + self.weights[1] + attack_matrix

        # Reshape back to the output tensor shape:
        z = z_col.reshape(batch_size, output_height, output_width, self.num_filters)

        # Apply activation function if specified
        if self.activation:
            return self.activation(z)
        return z

    

class Flatten:
    def forward(self, input_data, return_all_outputs, fixed_point, analysis=False):
        # Save the original shape for potential use in backward pass
        self.input_shape = input_data.shape
        # Flatten the input data
        return input_data.reshape(self.input_shape[0], -1), []
    

