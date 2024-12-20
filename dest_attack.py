import os
import numpy as np
import tensorflow as tf
import h5py
from network import Network  # Assuming Network class is in a separate file
from layers import Dense, Conv2D, Flatten
from activation import relu, softmax


# Load MNIST dataset (from your training code)
def load_data(dataset_name="mnist"):
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


def load_weights_and_biases(model, directory):
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights') and layer.weights:
            # Load weights
            weight_file = os.path.join(directory, f"layer_{i}_weights.npy")
            bias_file = os.path.join(directory, f"layer_{i}_biases.npy")
            layer.weights[0] = np.load(weight_file)
            layer.weights[1] = np.load(bias_file)
            print(layer.weights[0].size, layer.weights[1].size)


# Define the function to run inference on MNIST images with a specific label
def run_inference_on_label_batch(model, x_data, y_data, target_label, return_all_outputs=False):
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
    # Filter out samples with the specified label
    indices = np.where(y_data == target_label)[0]
    inputs = x_data[indices]  # Select all samples with the target label

    # Perform a forward pass for all samples at once
    if return_all_outputs:
        all_layer_outputs = model.forward(inputs, return_all_outputs=True)  # List of layer outputs
        # Compute average for each layer
        avg_outputs = [np.mean(layer_output, axis=0) for layer_output in all_layer_outputs]
        return avg_outputs
    else:
        final_output = model.forward(inputs, return_all_outputs=False)  # Shape: (num_samples, ...)
        return final_output


# Save the output of each layer to a dictionary or text file
def save_layer_outputs(model, input_data, target_label):
    layer_outputs = {}
    data = input_data
    
    for i, layer in enumerate(model.layers):
        output = layer.forward(data)  # Perform the forward pass for the layer
        layer_outputs[layer.__class__.__name__ + f'_{i}'] = output  # Store the output with layer name

    # Optionally save outputs to a file or use for further analysis
    # Example: Saving as a .npz file (could be adjusted as needed)
    np.savez(f"layer_outputs_label_{target_label}.npz", **layer_outputs)
    print(f"Layer outputs for label {target_label} saved.")


# Function to load and run inference on all models
def run_inference_on_all_models(models_info, target_label, return_all_outputs=False):
    """
    Run inference on all models and compute outputs for a specific label.

    Args:
        models_info (list): List of tuples containing model names and file paths.
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Corresponding labels.
        target_label (int): The label to run inference on.
        return_all_outputs (bool): Whether to return all layer outputs.

    Returns:
        None
    """
    for model_name, model_path in models_info:
        print(f"Loading model {model_name} from {model_path}")
        
        # Initialize the model with the correct architecture
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
            print(f"Model architecture for {model_name} is not defined!")
            continue

        # Load weights into the model
        load_weights_and_biases(model, model_path)

        # Choose the correct dataset based on the model
        if model_name in ["DNN_CIFAR10", "LeNet5"]:
            dataset_name = "cifar10"
        else:
            dataset_name = "mnist"

        (x_train, y_train), (x_test, y_test) = load_data(dataset_name)
        
        # Run inference on the specified label
        print(f"Running inference for {model_name} on label {target_label} images.")
        outputs = run_inference_on_label_batch(model, x_train, y_train, target_label, return_all_outputs)
        
        # Print the outputs
        print('Outputs: ', outputs)
        if return_all_outputs:
            for i, layer_output in enumerate(outputs):
                print(f"Average output for layer {i} of model {model_name}: {layer_output}")
        else:
            print(f"Average final output of model {model_name}: {outputs}")


# Main program to load data, create models, and run inference
def main():
    # Define model information
    models_info = [
        ("DNN_3_layers", "models/mnist/DNN_3_layers"),
        ("DNN_5_layers", "models/mnist/DNN_5_layers"),
        ("DNN_CIFAR10", "models/cifar10/DNN_CIFAR10")
    ]

    # Target label for inference
    target_label = 1  # Example: Change to any other label as needed

    # Run inference for all models, returning all layer outputs
    return_all_outputs = True
    run_inference_on_all_models(models_info, target_label, return_all_outputs)


if __name__ == '__main__':
    main()
