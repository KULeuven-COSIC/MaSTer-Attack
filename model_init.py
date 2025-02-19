from network import Network  
from layers import Dense, Conv2D, Flatten
from activation import relu, softmax, sigmoid, tanh
from tensorflow.keras import models, layers
from data_loader import DataLoader

LABEL_RANGES = {
        "MNIST": range(10),  # 0-9
        "CIFAR10": range(10),  # 0-9
        "MITBIH": range(5),  # 0-4 for heartbeat classes
        "VOICE": range(2),
        "OBESITY": range(7)
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
        elif model_name == "DNN_5_CIFAR10":
            model.add(Flatten())
            model.add(Dense(32 * 32 * 3, 512, activation=relu))
            model.add(Dense(512, 256, activation=relu))
            model.add(Dense(256, 256, activation=relu))
            model.add(Dense(256, 128, activation=relu))
            model.add(Dense(128, 10, activation=softmax))
        elif model_name == "LeNet5_CIFAR10":
            model.add(Conv2D(input_shape=(None, 32, 32, 3), filter_size=5, num_filters=6, activation=tanh))
            model.add(Conv2D(input_shape=(None, 28, 28, 6), filter_size=5, num_filters=16, activation=tanh))
            model.add(Flatten())
            model.add(Dense(24 * 24 * 16, 120, activation=tanh))
            model.add(Dense(120, 84, activation=tanh))
            model.add(Dense(84, 10, activation=softmax))
        elif model_name == "DNN_3_MITBIH":
            model.add(Dense(187, 50, activation=relu))
            model.add(Dense(50, 50, activation=relu))
            model.add(Dense(50, 5, activation=softmax))
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
    
    def get_tf_config():
        dataset_configs = [
        # {
        #     "dataset_name": "MNIST",
        #     "models_info": [
        #         ("DNN_3_MNIST", [
        #             layers.Dense(128, activation='relu', input_shape=(784,)),
        #             layers.Dense(64, activation='relu'),
        #             layers.Dense(10, activation='softmax')
        #         ]),
        #         ("DNN_5_MNIST", [
        #             layers.Dense(256, activation='relu', input_shape=(784,)),
        #             layers.Dense(128, activation='relu'),
        #             layers.Dense(64, activation='relu'),
        #             layers.Dense(32, activation='relu'),
        #             layers.Dense(10, activation='softmax')
        #         ]),
        #         ("DNN_7_MNIST", [
        #             layers.Dense(512, activation='relu', input_shape=(784,)),
        #             layers.Dense(256, activation='relu'),
        #             layers.Dense(128, activation='relu'),
        #             layers.Dense(64, activation='relu'),
        #             layers.Dense(32, activation='relu'),
        #             layers.Dense(16, activation='relu'),
        #             layers.Dense(10, activation='softmax')
        #         ])
        #     ],
        #     "data_loader": DataLoader.load_mnist_data,
        # },
        {
            "dataset_name": "CIFAR10",
            "models_info": [
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
                ("LeNet5_CIFAR10", [
                    layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 3)),
                    layers.Conv2D(16, (5, 5), activation='tanh'),
                    layers.Flatten(),
                    layers.Dense(120, activation='tanh'),
                    layers.Dense(84, activation='tanh'),
                    layers.Dense(10, activation='softmax')
                ]),
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
            ],
            "data_loader": DataLoader.load_cifar10_data,
        }
        # ,
        # {
        #     "dataset_name": "MITBIH",
        #     "models_info": [
        #         ("DNN_3_MITBIH", [
        #             layers.Dense(50, activation='relu'),
        #             layers.Dense(50, activation='relu'),
        #             layers.Dense(5, activation='softmax'),
        #         ]),
        #         ("DNN_5_MITBIH", [
        #             layers.Dense(50, activation='relu'),
        #             layers.Dense(50, activation='relu'),
        #             layers.Dense(50, activation='relu'),
        #             layers.Dense(50, activation='relu'),
        #             layers.Dense(5, activation='softmax'),
        #         ])
        #     ],
        #     "data_loader": DataLoader.load_mitbih_data,
        # },
        # {
        #     "dataset_name": "VOICE",
        #     "models_info": [
        #         ("DNN_5_VOICE", [
        #             layers.Dense(256, activation='relu'),
        #             layers.Dense(128, activation='relu'),
        #             layers.Dense(128, activation='relu'),
        #             layers.Dense(64, activation='relu'),
        #             layers.Dense(1, activation='sigmoid'),
        #         ])
        #     ],
        #     "data_loader": DataLoader.load_voice_data,
        # },
        # {
        #     "dataset_name": "OBESITY",
        #     "models_info": [
        #         ("DNN_5_OBESITY", [
        #             layers.Dense(128, activation='relu'),
        #             layers.Dense(64, activation='relu'),
        #             layers.Dense(64, activation='relu'),
        #             layers.Dense(32, activation='relu'),
        #             layers.Dense(7, activation='softmax'),
        #         ])
        #     ],
        #     "data_loader": DataLoader.load_obesity_data,
        # }
        ]
        return dataset_configs
        