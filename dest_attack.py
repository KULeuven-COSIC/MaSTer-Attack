import os
import numpy as np
import tensorflow as tf
import h5py
from network import Network  # Assuming Network class is in a separate file
from layers import Dense, Conv2D, Flatten
from activation import relu, softmax

# Utilities for loading data
class DataLoader:
    @staticmethod
    def load_data(dataset_name="mnist"):
        """
        Load the specified dataset (MNIST or CIFAR-10).

        Args:
            dataset_name (str): Name of the dataset to load ('mnist' or 'cifar10').

        Returns:
            tuple: Training and testing datasets (x_train, y_train), (x_test, y_test).
        """
        if dataset_name == "mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255
            x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255
        elif dataset_name == "cifar10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255  # Normalize CIFAR-10 data
            x_test = x_test.astype('float32') / 255
        else:
            raise ValueError("Unsupported dataset. Please choose 'mnist' or 'cifar10'.")
        return (x_train, y_train), (x_test, y_test)

# Utilities for loading weights
class WeightLoader:
    @staticmethod
    def load_weights_and_biases(model, directory):
        """
        Load weights and biases for the given model from the specified directory.

        Args:
            model (Network): The neural network model.
            directory (str): Path to the directory containing weight files.
        """
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'weights') and layer.weights:
                weight_file = os.path.join(directory, f"layer_{i}_weights.npy")
                bias_file = os.path.join(directory, f"layer_{i}_biases.npy")
                layer.weights[0] = np.load(weight_file)
                layer.weights[1] = np.load(bias_file)
                # print(f"Loaded weights and biases for layer {i}: {layer.weights[0].size}, {layer.weights[1].size}")

# Utilities for model initialization
class ModelInitializer:
    @staticmethod
    def initialize_model(model_name):
        """
        Initialize a model based on its name.

        Args:
            model_name (str): Name of the model to initialize.

        Returns:
            Network: An instance of the initialized model.
        """
        model = Network()
        if model_name == "DNN_3_layers":
            model.add(Dense(784, 128, activation=relu))
            model.add(Dense(128, 64, activation=relu))
            model.add(Dense(64, 10, activation=softmax))
        elif model_name == "DNN_5_layers":
            model.add(Dense(784, 256, activation=relu))
            model.add(Dense(256, 128, activation=relu))
            model.add(Dense(128, 64, activation=relu))
            model.add(Dense(64, 32, activation=relu))
            model.add(Dense(32, 10, activation=softmax))
        elif model_name == "DNN_CIFAR10":
            model.add(Flatten())
            model.add(Dense(32 * 32 * 3, 512, activation=relu))
            model.add(Dense(512, 256, activation=relu))
            model.add(Dense(256, 10, activation=softmax))
        elif model_name == "LeNet5":
            model.add(Conv2D(input_shape=(32, 32, 3), filter_size=5, num_filters=6, activation=relu))
            model.add(Conv2D(input_shape=(28, 28, 6), filter_size=5, num_filters=16, activation=relu))
            model.add(Flatten())
            model.add(Dense(400, 120, activation=relu))
            model.add(Dense(120, 84, activation=relu))
            model.add(Dense(84, 10, activation=softmax))
        else:
            raise ValueError(f"Model architecture for {model_name} is not defined!")
        return model

# Inference runner
class InferenceRunner:
    @staticmethod
    def run_inference_on_label_batch(model, x_data, y_data, target_label, return_all_outputs, fixed_point):
        """
        Run inference on all samples of a specific label and compute average outputs.

        Args:
            model (Network): The neural network model.
            x_data (np.ndarray): Input data.
            y_data (np.ndarray): Corresponding labels.
            target_label (int): The label for which inference is performed.
            return_all_outputs (bool): Whether to return outputs for all layers.

        Returns:
            np.ndarray: Average outputs of each layer if return_all_outputs is True.
            np.ndarray: Average final output if return_all_outputs is False.
        """
        indices = np.where(y_data == target_label)[0]
        inputs = x_data[indices]

        if return_all_outputs:
            all_layer_outputs = model.forward(inputs, return_all_outputs=True, fixed_point=fixed_point)
            avg_outputs = [np.mean(layer_output, axis=0) for layer_output in all_layer_outputs]
            return avg_outputs
        else:
            return model.forward(inputs, return_all_outputs=False, fixed_point=fixed_point)

    @staticmethod
    def run_inference_on_all_models(models_info, target_label, return_all_outputs=False, fixed_point=None):
        """
        Run inference on all models and compute outputs for a specific label.

        Args:
            models_info (list): List of tuples containing model names and file paths.
            target_label (int): The label to run inference on.
            return_all_outputs (bool): Whether to return all layer outputs.

        Returns:
            dict: Dictionary with model names as keys and their outputs as values.
        """
        results = {}

        for model_name, model_path in models_info:
            print(f"Loading model {model_name} from {model_path}")

            model = ModelInitializer.initialize_model(model_name)
            WeightLoader.load_weights_and_biases(model, model_path)

            dataset_name = "cifar10" if model_name in ["DNN_CIFAR10", "LeNet5"] else "mnist"
            (x_train, y_train), _ = DataLoader.load_data(dataset_name)

            print(f"Running inference for {model_name} on label {target_label} images.")
            outputs = InferenceRunner.run_inference_on_label_batch(model, x_train, y_train, target_label, return_all_outputs, fixed_point)
            results[model_name] = outputs

        return results

class AttackRunner:
    @staticmethod
    def run_attack_on_all_models(models_info, target_label, return_all_outputs, attack_type, attack_reference, fixed_point):
        """
        Run attack on all models and compute outputs for a specific label.

        Args:
            models_info (list): List of tuples containing model names and file paths.
            target_label (int): The label to run attack on.
            return_all_outputs (bool): Whether to return all layer outputs.
            attack_type (str): Type of attack to perform.
            attack_reference (dict): Reference outputs for attack computation.

        Returns:
            dict: Dictionary with model names as keys and their attack results as values.
        """
        results = {}

        for model_name, model_path in models_info:
            print(f"Running attack on model {model_name} for label {target_label} with fixed-point precision {fixed_point}")

            model = ModelInitializer.initialize_model(model_name)
            WeightLoader.load_weights_and_biases(model, model_path)

            dataset_name = "cifar10" if model_name in ["DNN_CIFAR10", "LeNet5"] else "mnist"
            (x_train, y_train), _ = DataLoader.load_data(dataset_name)

            indices = np.where(y_train != target_label)[0]
            inputs = x_train[indices]

            outputs = model.forward_attack(inputs, return_all_outputs, attack_type, attack_reference[model_name], fixed_point)
            results[model_name] = outputs
            1/0

        return results

# Main function
if __name__ == '__main__':
    models_info = [
        ("DNN_3_layers", "models/mnist/DNN_3_layers"),
        ("DNN_5_layers", "models/mnist/DNN_5_layers"),
        ("DNN_CIFAR10", "models/cifar10/DNN_CIFAR10")
    ]

    reference_matrices = {}
    for target_label in range(10):
        print(f"Running inference for label {target_label}")
        reference_matrices[f"reference_matrices_{target_label}"] = InferenceRunner.run_inference_on_all_models(models_info, target_label, return_all_outputs=True)

    attack_rate = {}
    attack_type = "layer_output_matching"

    for target_label in range(10):
        print(f"Running attack for label {target_label}")
        attack_rate[f"attack_rate_{target_label}"] = AttackRunner.run_attack_on_all_models(models_info, target_label, return_all_outputs=False, attack_type=attack_type, attack_reference=reference_matrices[f"reference_matrices_{target_label}"], fixed_point = 8 )
    print(attack_rate[f"attack_rate_1"]['DNN_5_layers'][:10])
    print(attack_rate[f"attack_rate_1"]['DNN_CIFAR10'][:10])


# THE RETURN_ALL_OUTPUTS RETURNS OUTPUTS OF EACH LAYER AFTER ACTIVATION - NEED TO CHANGE THIS !!!