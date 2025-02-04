import tensorflow as tf
import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

    def load_mnist_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255
        x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255
        return x_train, y_train, x_test, y_test

    def load_cifar10_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        y_train = y_train.flatten()  # Flatten labels
        y_test = y_test.flatten()  # Flatten labels
        return x_train, y_train, x_test, y_test

    def load_mitbih_data():
        mit_test_data = pd.read_csv('data/mitbih_test.csv', header=None)
        mit_train_data = pd.read_csv('data/mitbih_train.csv', header=None)

        x_train, y_train = mit_train_data.iloc[: , :-1], mit_train_data.iloc[: , -1]
        x_train, valX, y_train, valy= train_test_split(x_train,y_train,test_size=0.2)
        testX, testy = mit_test_data.iloc[: , :-1], mit_test_data.iloc[: , -1]
        y_train = to_categorical(y_train)
        testy = to_categorical(testy)
        return x_train, y_train, testX, testy

    def load_voice_data():
        dataframe = pd.read_csv('data/voice.csv')
        dict = {'label':{'male':1,'female':0}}  
        dataframe.replace(dict,inplace = True)        
        x_train = dataframe.loc[:, dataframe.columns != 'label']
        y_train = dataframe.loc[:,'label']
        return x_train, y_train

    def load_obesity_data():
        df = pd.read_csv('data/Obesity prediction.csv')

        # Initialize label encoders and store them in a dictionary
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

        x_train = df.drop('Obesity', axis=1)
        y_train = df['Obesity']
        return x_train, y_train