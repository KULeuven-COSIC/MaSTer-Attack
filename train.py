import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

# Function to save model weights and biases to a text file
def save_weights_to_txt(model, directory):
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if weights:  # Only process layers that have weights
            for j, weight in enumerate(weights):
                weight_file_path = os.path.join(directory, f"{i}_weights_{j}.txt")
                # Check the dimensions of the weight array
                if weight.ndim == 4:  # 4D for Conv layers
                    # Save each slice of the 4D weight tensor separately
                    for j in range(weight.shape[3]):  # Iterate over the number of filters
                        slice_file_path = os.path.join(directory, f"{layer.name}_weights_{i}_filter_{j}.txt")
                        np.savetxt(slice_file_path, weight[..., j].reshape(-1), fmt='%.6f')
                else:
                    # Save weights normally for 1D or 2D arrays
                    np.savetxt(weight_file_path, weight, fmt='%.6f')

def save_weights_and_biases(model, directory):
    os.makedirs(directory, exist_ok=True)
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights') and layer.weights:
            # Save weights
            np.save(os.path.join(directory, f"layer_{i}_weights.npy"), layer.weights[0])
            # Save biases
            np.save(os.path.join(directory, f"layer_{i}_biases.npy"), layer.weights[1])

# Function to train and save DNNs on MNIST
def train_mnist_models():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255

    # Define models
    models_info = [
        ("DNN_3_layers", [
            layers.Dense(128, activation='relu', input_shape=(784,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ]),
        ("DNN_5_layers", [
            layers.Dense(256, activation='relu', input_shape=(784,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
    ]

    # Train and save each model
    for model_name, layer_list in models_info:
        model_dir = f"models/mnist/{model_name}"
        os.makedirs(model_dir, exist_ok=True)

        model = models.Sequential(layer_list)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

        # Save the whole model
        model.save(os.path.join(model_dir, f"{model_name}.h5"))

        # Save weights and biases to text files
        save_weights_and_biases(model, model_dir)

# Function to create and train CNNs on CIFAR-10
def train_cifar10_models():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = y_train.flatten()  # Flatten labels
    y_test = y_test.flatten()  # Flatten labels

    # Define models
    models_info = [
        ("DNN_CIFAR10", [
            layers.Flatten(input_shape=(32, 32, 3)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(10, activation='softmax')
        ]),
        ("LeNet5", [
            layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 3)),
            layers.Conv2D(16, (5, 5), activation='tanh'),
            layers.Flatten(),
            layers.Dense(120, activation='tanh'),
            layers.Dense(84, activation='tanh'),
            layers.Dense(10, activation='softmax')
        ])
        # ,
        # ("AlexNet", [
        #     layers.Conv2D(96, (5, 5), activation='relu', input_shape=(32, 32, 3), strides=1, padding='same'),
        #     layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'),
        #     layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
        #     layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'),
        #     layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
        #     layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
        #     layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        #     layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'),
        #     layers.Flatten(),
        #     layers.Dense(4096, activation='relu'),
        #     layers.Dropout(0.5),
        #     layers.Dense(4096, activation='relu'),
        #     layers.Dropout(0.5),
        #     layers.Dense(10, activation='softmax')
        # ])
    ]

    # Train and save each model
    for model_name, layer_list in models_info:
        model_dir = f"models/cifar10/{model_name}"
        os.makedirs(model_dir, exist_ok=True)

        model = models.Sequential(layer_list)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

        # Save the whole model
        model.save(os.path.join(model_dir, f"{model_name}.h5"))

        # Save weights and biases to text files
        save_weights_and_biases(model, model_dir)

if __name__ == "__main__":
    train_mnist_models()
    train_cifar10_models()
