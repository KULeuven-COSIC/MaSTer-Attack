import numpy as np
import tensorflow as tf
from network import Network
from layers import Dense
from layers import Conv2D
from layers import Flatten
from activation import relu, softmax
from tensorflow.keras import layers, models

def train_and_save_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data: flatten images and normalize to [0, 1]
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255

    # Build the model
    model = models.Sequential([
        layers.Dense(300, activation='relu', input_shape=(784,)),  # Specify input shape for the first layer
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

    # Save the model weights
    model.save('mnist_dnn_model.h5')

# Load the pre-trained TensorFlow model
def load_tf_model():
    return tf.keras.models.load_model('mnist_dnn_model.h5')

# Create the custom DNN model
def create_custom_model():
    nn = Network()
    nn.add(Dense(input_size=784, output_size=300, activation=relu))  # First dense layer
    nn.add(Dense(input_size=300, output_size=10, activation=softmax))  # Output layer
    return nn

# Load weights from TensorFlow model into custom model
def load_weights_into_custom_model(custom_model, tf_model):
    dense_layer1 = tf_model.layers[0]
    dense_layer2 = tf_model.layers[1]
    
    # Load weights into the custom dense layers
    custom_dense_layer1 = custom_model.layers[0]
    custom_dense_layer1.weights[0] = dense_layer1.get_weights()[0]  # Weights
    custom_dense_layer1.weights[1] = dense_layer1.get_weights()[1]  # Biases

    custom_dense_layer2 = custom_model.layers[1]
    custom_dense_layer2.weights[0] = dense_layer2.get_weights()[0]  # Weights
    custom_dense_layer2.weights[1] = dense_layer2.get_weights()[1]  # Biases

# Run inference with the TensorFlow model
def run_inference_with_tf_model(model, input_data):
    return model.predict(input_data)

# Run inference with the custom model
def run_inference_with_custom_model(custom_model, input_data):
    return custom_model.forward(input_data)

# Test model inference
def test_model_inference():
    # Train and save the model
    train_and_save_model()

    # Load the TensorFlow model
    tf_model = load_tf_model()

    # Create the custom neural network model
    custom_model = create_custom_model()

    # Load the weights into the custom model
    load_weights_into_custom_model(custom_model, tf_model)

    # Test with some samples from the MNIST test set
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255  # Flatten the test set

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