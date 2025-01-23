from network import Network  
from layers import Dense, Conv2D, Flatten
from activation import relu, softmax, sigmoid

LABEL_RANGES = {
        "mnist": range(10),  # 0-9
        "cifar10": range(10),  # 0-9
        "mitbih": range(5),  # 0-4 for heartbeat classes
        "voice": range(2),
        "obesity": range(7)
    }

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
        if model_name == "DNN_3_MNIST":
            model.add(Dense(784, 128, activation=relu))
            model.add(Dense(128, 64, activation=relu))
            model.add(Dense(64, 10, activation=softmax))
        elif model_name == "DNN_5_MNIST":
            model.add(Dense(784, 256, activation=relu))
            model.add(Dense(256, 128, activation=relu))
            model.add(Dense(128, 64, activation=relu))
            model.add(Dense(64, 32, activation=relu))
            model.add(Dense(32, 10, activation=softmax))
        elif model_name == "DNN_7_MNIST":
            model.add(Dense(784, 512, activation=relu))
            model.add(Dense(512, 256, activation=relu))
            model.add(Dense(256, 128, activation=relu))
            model.add(Dense(128, 64, activation=relu))
            model.add(Dense(64, 32, activation=relu))
            model.add(Dense(32, 16, activation=relu))
            model.add(Dense(16, 10, activation=softmax))
        elif model_name == "DNN_3_CIFAR10":
            model.add(Flatten())
            model.add(Dense(32 * 32 * 3, 512, activation=relu))
            model.add(Dense(512, 256, activation=relu))
            model.add(Dense(256, 10, activation=softmax))
        elif model_name == "LeNet5_CIFAR10":
            model.add(Conv2D(input_shape=(32, 32, 3), filter_size=5, num_filters=6, activation=relu))
            model.add(Conv2D(input_shape=(28, 28, 6), filter_size=5, num_filters=16, activation=relu))
            model.add(Flatten())
            model.add(Dense(400, 120, activation=relu))
            model.add(Dense(120, 84, activation=relu))
            model.add(Dense(84, 10, activation=softmax))
        elif model_name == "DNN_5_MITBIH":
            model.add(Dense(187, 50, activation=relu))
            model.add(Dense(50, 50, activation=relu))
            model.add(Dense(50, 50, activation=relu))
            model.add(Dense(50, 50, activation=relu))
            model.add(Dense(50, 5, activation=softmax))
        elif model_name == "DNN_5_VOICE":
            model.add(Dense(20, 256, activation=relu))
            model.add(Dense(256, 128, activation=relu))
            model.add(Dense(128, 128, activation=relu))
            model.add(Dense(128, 64, activation=relu))
            model.add(Dense(64, 1, activation=sigmoid))
        elif model_name == "DNN_5_OBESITY":
            model.add(Dense(16, 128, activation=relu))
            model.add(Dense(128, 64, activation=relu))
            model.add(Dense(64, 64, activation=relu))
            model.add(Dense(64, 32, activation=relu))
            model.add(Dense(64, 7, activation=softmax))
        else:
            raise ValueError(f"Model architecture for {model_name} is not defined!")
        return model