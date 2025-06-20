import numpy as np
import tensorflow as tf
from network import Network
from layers import Dense
from layers import Conv2D
from layers import Flatten
from activation import relu, softmax, tanh

from tensorflow.keras import layers, models

def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255
    # y_train = y_train.flatten()  # Flatten labels
    # y_test = y_test.flatten()  # Flatten labels

    # Build the model
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 3)),
        layers.Conv2D(16, (5, 5), activation='tanh'),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

    # Save the model weights
    model.save('mnist_cnn_model.h5')


def load_tf_model():
    # Load the pre-trained TensorFlow model
    return tf.keras.models.load_model('mnist_cnn_model.h5')

def create_custom_model():
    # Create a custom neural network
    nn = Network()
    nn.add(Conv2D(input_shape=(None, 32, 32, 3), filter_size=5, num_filters=6, activation=tanh))
    nn.add(Conv2D(input_shape=(None, 28, 28, 6), filter_size=5, num_filters=16, activation=tanh))
    nn.add(Flatten())
    nn.add(Dense(input_size=24 * 24 * 16, output_size=120, activation=tanh))
    nn.add(Dense(input_size=120, output_size=84, activation=tanh))
    nn.add(Dense(input_size=84, output_size=10, activation=softmax))
    return nn

def load_weights_into_custom_model(custom_model, tf_model):
    # Load weights from TensorFlow model into the custom model
    conv_layer1 = tf_model.layers[0]
    conv_layer2 = tf_model.layers[1]
    dense_layer1 = tf_model.layers[3]
    dense_layer2 = tf_model.layers[4]
    dense_layer3 = tf_model.layers[5]
    
    # Load Conv2D weights
    custom_conv_layer1 = custom_model.layers[0]
    custom_conv_layer1.weights[0] = conv_layer1.get_weights()[0]  # Conv weights
    custom_conv_layer1.weights[1] = conv_layer1.get_weights()[1]  # Conv biases

    # Load Conv2D weights
    custom_conv_layer2 = custom_model.layers[1]
    custom_conv_layer2.weights[0] = conv_layer2.get_weights()[0]  # Conv weights
    custom_conv_layer2.weights[1] = conv_layer2.get_weights()[1]  # Conv biases
    
    # Load Dense weights
    custom_dense_layer1 = custom_model.layers[3]
    custom_dense_layer1.weights[0] = dense_layer1.get_weights()[0]  # Dense weights
    custom_dense_layer1.weights[1] = dense_layer1.get_weights()[1]  # Dense biases

    # Load Dense weights
    custom_dense_layer2 = custom_model.layers[4]
    custom_dense_layer2.weights[0] = dense_layer2.get_weights()[0]  # Dense weights
    custom_dense_layer2.weights[1] = dense_layer2.get_weights()[1]  # Dense biases

    # Load Dense weights
    custom_dense_layer3 = custom_model.layers[5]
    custom_dense_layer3.weights[0] = dense_layer3.get_weights()[0]  # Dense weights
    custom_dense_layer3.weights[1] = dense_layer3.get_weights()[1]  # Dense biases

def run_inference_with_tf_model(model, input_data):
    return model.predict(input_data)

def run_inference_with_custom_model(custom_model, input_data):
    return custom_model.forward(input_data)

def test_model_inference():
    # Train and save the model
    train_and_save_model()

    # Load the TensorFlow model
    tf_model = load_tf_model()

    # Create the custom neural network model
    custom_model = create_custom_model()

    # Load the weights into the custom model
    load_weights_into_custom_model(custom_model, tf_model)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255

    # Use the first few samples for testing
    test_samples = x_test[:10]

    # Get predictions from TensorFlow model
    tf_outputs = run_inference_with_tf_model(tf_model, test_samples)

    # Get predictions from custom neural network model
    custom_outputs = run_inference_with_custom_model(custom_model, test_samples)

    # Compare the outputs
    print("TensorFlow outputs:\n", tf_outputs)
    print("Custom model outputs:\n", custom_outputs)

    # Check if the outputs are similar (allowing for small numerical differences)
    are_outputs_similar = np.allclose(tf_outputs, custom_outputs, rtol=1e-5)
    
    # Assert that the outputs are similar and raise an error with a message if not
    assert are_outputs_similar, "The outputs from the TensorFlow model and the custom model are not similar!"

if __name__ == "__main__":
    test_model_inference()
