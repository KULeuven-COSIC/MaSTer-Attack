import os
import numpy as np
import pandas as pd 
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from data_loader import DataLoader
from model_init import ModelInitializer
from visualiser import Visualise

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

def save_correctly_classified_samples(model, x_data, y_data, predictions, model_name, dataset_dir="datasets"):
    # Get the predicted labels
    if model_name == 'DNN_5_VOICE':
        print(predictions)
        print(np.argmax(predictions, axis=1))
    predicted_labels = np.argmax(predictions, axis=1)

    # Ensure the dataset directory exists
    os.makedirs(f"{dataset_dir}/{model_name}", exist_ok=True)

    # Process each label (0 to 4 in the MITBIH dataset)
    for label in np.unique(y_data):
        # Identify samples where the true label matches the predicted label
        correct_classification = (y_data == label) & (predicted_labels == label)
        
        # Get the correctly classified samples
        correctly_classified_samples = x_data[correct_classification]
        
        # If there are correctly classified samples, save them to a CSV file
        if correctly_classified_samples.shape[0] > 0:
            label_dir = f"{dataset_dir}/{model_name}/label_{label}"
            os.makedirs(label_dir, exist_ok=True)

            # Dynamically flatten if the data is more than 2D
            if model_name != 'LeNet5_CIFAR10':
                if len(correctly_classified_samples.shape) > 2:  # More than 2D
                    print(model_name, correctly_classified_samples.shape)
                    flat_samples = correctly_classified_samples.reshape(correctly_classified_samples.shape[0], -1)
                else:
                    flat_samples = correctly_classified_samples  # Already 2D
            else:
                flat_samples = correctly_classified_samples

            # Save these samples as a .npy file
            sample_file = os.path.join(label_dir, f"label_{label}_correct.npy")
            np.save(sample_file, flat_samples)


# Function to train and save DNNs on MNIST
def train_mnist_models():
    # Load MNIST dataset
    x_train, y_train, x_test, y_test = DataLoader.load_mnist_data()

    # Define models
    models_info = [
        ("DNN_3_MNIST", [
            layers.Dense(128, activation='relu', input_shape=(784,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ]),
        ("DNN_5_MNIST", [
            layers.Dense(256, activation='relu', input_shape=(784,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')
        ]),
        ("DNN_7_MNIST", [
            layers.Dense(512, activation='relu', input_shape=(784,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
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

        # Make predictions on the test set
        predictions = model.predict(x_test)

        # Save correctly classified samples for each label
        save_correctly_classified_samples(model, x_test, y_test, predictions, model_name)

        # Save the whole model
        model.save(os.path.join(model_dir, f"{model_name}.h5"))

        # Save weights and biases to text files
        save_weights_and_biases(model, model_dir)

# Function to create and train CNNs on CIFAR-10
def train_cifar10_models():
    # Load CIFAR-10 dataset
    x_train, y_train, x_test, y_test = DataLoader.load_cifar10_data()

    # Define models
    models_info = [
        ("DNN_3_CIFAR10", [
            layers.Flatten(input_shape=(32, 32, 3)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(10, activation='softmax')
        ]),
        ("DNN_5_CIFAR10", [
            layers.Flatten(input_shape=(32, 32, 3)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ]),
        # ("LeNet5_CIFAR10", [
        #     layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 3)),
        #     layers.Conv2D(16, (5, 5), activation='tanh'),
        #     layers.Flatten(),
        #     layers.Dense(120, activation='tanh'),
        #     layers.Dense(84, activation='tanh'),
        #     layers.Dense(10, activation='softmax')
        # ]),
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
        model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.1)

        # Make predictions on the test set
        predictions = model.predict(x_test)

        # Save correctly classified samples for each label
        save_correctly_classified_samples(model, x_test, y_test, predictions, model_name)

        # Save the whole model
        model.save(os.path.join(model_dir, f"{model_name}.h5"))

        # Save weights and biases to text files
        save_weights_and_biases(model, model_dir)
        

def train_mitbih_models():
    x_train, y_train, testX, testy = DataLoader.load_mitbih_data()
    
    # Define models
    models_info = [
        ("DNN_5_MITBIH", [
            layers.Dense(50, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(5, activation='softmax'),
        ])
    ]

    # Train and save each model
    for model_name, layer_list in models_info:
        model_dir = f"models/mitbih/{model_name}"
        os.makedirs(model_dir, exist_ok=True)

        model = models.Sequential(layer_list)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(valX, valy))

         # Make predictions on the test set
        predictions = model.predict(testX)
        predicted_labels = np.argmax(predictions, axis=1)  # Get predicted labels
        if len(testy.shape) > 1 and testy.shape[1] > 1:
            true_labels = np.argmax(testy, axis=1)  # Convert to single-label format
        else:
            true_labels = testy
        
        cm = confusion_matrix(true_labels, predicted_labels)
    
        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.arange(5), yticklabels=np.arange(5))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # Identify misclassifications where the true label is not 0 but predicted label is 0
        misclassified_to_zero = (true_labels != 0) & (predicted_labels == 0)
        total_misclassified_to_zero = np.sum(misclassified_to_zero)
        
        # Calculate the percentage
        total_non_zero_labels = np.sum(true_labels != 0)
        percentage_misclassified_to_zero = (total_misclassified_to_zero / total_non_zero_labels) * 100
        
        
        print(f"Percentage of non-zero labels misclassified as 0: {percentage_misclassified_to_zero:.2f}%")

        testy =  np.argmax(testy, axis=1)

        # Save correctly classified samples for each label
        save_correctly_classified_samples(model, testX, testy, predictions, model_name)

        # Save the whole model
        model.save(os.path.join(model_dir, f"{model_name}.h5"))

        # Save weights and biases to text files
        save_weights_and_biases(model, model_dir)

def train_voice_models():
    x_train, y_train = DataLoader.load_voice_data()

    # X = dataframe.drop('label',axis=1)
    # y = dataframe[['label']]
    
    # Define models
    models_info = [
        ("DNN_5_VOICE", [
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])
    ]

    # Train and save each model
    for model_name, layer_list in models_info:
        model_dir = f"models/voice/{model_name}"
        os.makedirs(model_dir, exist_ok=True)

        model = models.Sequential(layer_list)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.1)

        # Make predictions on the test set
        predictions = model.predict(x_train)

        # Convert predictions to binary (0 or 1)
        binary_predictions = (predictions > 0.5).astype(int)

        # Convert binary predictions to one-hot encoding
        one_hot_predictions = np.zeros((binary_predictions.shape[0], 2))
        one_hot_predictions[np.arange(binary_predictions.shape[0]), binary_predictions.flatten()] = 1

        # Save correctly classified samples for each label
        save_correctly_classified_samples(model, x_train, y_train, one_hot_predictions, model_name)

        # Save the whole model
        model.save(os.path.join(model_dir, f"{model_name}.h5"))

        # Save weights and biases to text files
        save_weights_and_biases(model, model_dir)

def train_obesity_models():
    x_train, y_train = DataLoader.load_obesity_data()
    
    # Define models
    models_info = [
        ("DNN_5_OBESITY", [
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(7, activation='softmax'),
        ])
    ]

    # Train and save each model
    for model_name, layer_list in models_info:
        model_dir = f"models/obesity/{model_name}"
        os.makedirs(model_dir, exist_ok=True)

        model = models.Sequential(layer_list)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

        # Make predictions on the test set
        predictions = model.predict(x_train)

        # Save correctly classified samples for each label
        save_correctly_classified_samples(model, x_train, y_train, predictions, model_name)

        # Save the whole model
        model.save(os.path.join(model_dir, f"{model_name}.h5"))

        # Save weights and biases to text files
        save_weights_and_biases(model, model_dir)

def train_all_models():
    dataset_configs = ModelInitializer.get_tf_config()

    for config in dataset_configs:
        dataset_name = config['dataset_name']
        data_loader = config['data_loader']
        models_info = config['models_info']

        # Load dataset
        data = data_loader()
        if len(data) == 4:
            x_train, y_train, x_test, y_test = data
        elif len(data) == 2:
            x_train, y_train = data
            x_test, y_test = None, None

        # Train and save each model
        for model_name, layer_list in models_info:
            print(f'Training {model_name}')
            model_dir = f"models/{dataset_name.lower()}/{model_name}"
            os.makedirs(model_dir, exist_ok=True)

            model = models.Sequential(layer_list)

            # Determine loss function based on dataset and output
            if dataset_name == "VOICE":
                loss_function = 'binary_crossentropy'
            elif dataset_name == "MITBIH":
                loss_function = 'categorical_crossentropy'
            else:
                loss_function = 'sparse_categorical_crossentropy'

            model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

            # Adjust validation split or data for datasets without test sets
            if x_test is None:
                model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.1)
            else:
                model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))

            # Make predictions
            predictions = model.predict(x_train if x_test is None else x_test)

            if dataset_name == 'MITBIH':
                y_test_argmax =  np.argmax(y_test, axis=1)
            else:
                y_test_argmax = y_test

            if dataset_name == "VOICE":
                # Convert predictions to binary (0 or 1)
                binary_predictions = (predictions > 0.5).astype(int)

                # Convert binary predictions to one-hot encoding
                predictions = np.zeros((binary_predictions.shape[0], 2))
                predictions[np.arange(binary_predictions.shape[0]), binary_predictions.flatten()] = 1

            # Save correctly classified samples for each label
            save_correctly_classified_samples(model, x_train if x_test is None else x_test, y_train if y_test_argmax is None else y_test_argmax, predictions, model_name)

            # Save the whole model
            model.save(os.path.join(model_dir, f"{model_name}.h5"))

            # Save weights and biases to text files
            save_weights_and_biases(model, model_dir)

            # Get statistics on the values inside the model
            model = ModelInitializer.initialize_model(model_name)
            DataLoader.load_weights_and_biases(model, model_dir)
            aggregated_stats = model.forward(x_train if x_test is None else x_test, return_all_outputs=True, fixed_point=16, analysis=True)[1]
            print(f"Stats for {model_name}: {aggregated_stats['sign_stats']} and {aggregated_stats['truncated_stats']}")
            Visualise.plot_fixed_point_distribution(aggregated_stats['agg_trunc'], save_dir=f'analysis_plots/{dataset_name}/{model_name}')
            Visualise.plot_sign_distribution(aggregated_stats['sign_stats'], save_dir=f"analysis_plots/{dataset_name}/{model_name}")
            # Additional analysis for MITBIH dataset
            if dataset_name == "MITBIH" and x_test is not None:
                predicted_labels = np.argmax(predictions, axis=1)
                if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                    true_labels = np.argmax(y_test, axis=1)
                else:
                    true_labels = y_test

                cm = confusion_matrix(true_labels, predicted_labels)

                # Plot confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.arange(5), yticklabels=np.arange(5))
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.show()

                # Calculate percentage of misclassifications
                misclassified_to_zero = (true_labels != 0) & (predicted_labels == 0)
                total_misclassified_to_zero = np.sum(misclassified_to_zero)
                total_non_zero_labels = np.sum(true_labels != 0)
                percentage_misclassified_to_zero = (total_misclassified_to_zero / total_non_zero_labels) * 100

                print(f"Percentage of non-zero labels misclassified as 0: {percentage_misclassified_to_zero:.2f}%")

# if __name__ == "__main__":
#     train_all_models()
    # train_mnist_models()
    # train_cifar10_models()
    # train_mitbih_models()
    # train_voice_models()
    # train_obesity_models()
