import numpy as np
from data_loader import DataLoader
from model_init import ModelInitializer

class InferenceRunner:
    @staticmethod
    def run_inference_on_label_batch(model, label_data_file, return_all_outputs, fixed_point):
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
        

        inputs = np.load(label_data_file)

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

        model_name, model_path, dataset_name = models_info

        model = ModelInitializer.initialize_model(model_name)
        DataLoader.load_weights_and_biases(model, model_path)

        label_data_file = f"datasets/{model_name}/label_{target_label}/label_{target_label}_correct.npy"

        # if dataset_name == "mnist" or dataset_name == "cifar10":
        #     (X, Y), _ = DataLoader.load_data(dataset_name)
        # elif dataset_name == "mitbih":
        #     mit_test_data = pd.read_csv('data/mitbih_test.csv', header=None)
        #     X, Y = mit_test_data.iloc[: , :-1], mit_test_data.iloc[: , -1]
        #     Y = to_categorical(Y)
        #     Y = np.argmax(Y, axis=1)
        # elif dataset_name == "voice":
        #     dataframe = pd.read_csv('data/voice.csv')

        #     dict = {'label':{'male':1,'female':0}}  
        #     dataframe.replace(dict,inplace = True)        
        #     X = dataframe.loc[:, dataframe.columns != 'label']
        #     Y = dataframe.loc[:,'label']
        # elif dataset_name == "obesity":
        #     df = pd.read_csv('data/Obesity prediction.csv')

        #     # Initialize label encoders and store them in a dictionary
        #     label_encoders = {}
        #     for column in df.select_dtypes(include=['object']).columns:
        #         label_encoders[column] = LabelEncoder()
        #         df[column] = label_encoders[column].fit_transform(df[column])

        #     X = df.drop('Obesity', axis=1)
        #     Y = df['Obesity']

        print(f"Running inference for {model_name} on label {target_label} images.")
        outputs = InferenceRunner.run_inference_on_label_batch(model, label_data_file, return_all_outputs, fixed_point)

        return outputs