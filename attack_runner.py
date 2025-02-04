import numpy as np
from data_loader import DataLoader
from model_init import ModelInitializer
from model_init import LABEL_RANGES

class AttackRunner:
    @staticmethod
    def run_attack_on_all_models(models_info, target_label, return_all_outputs, attack_type, attack_reference, fixed_point, optimised):
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

        model_name, model_path, dataset_name = models_info

        print(f"Running attack on model {model_name} for label {target_label} with fixed-point precision {fixed_point}")

        model = ModelInitializer.initialize_model(model_name)
        DataLoader.load_weights_and_biases(model, model_path)

        if attack_type == "layer_output_matching":
            label_files = [
                f"datasets/{model_name}/label_{label}/label_{label}_correct.npy"
                for label in range(len(LABEL_RANGES[dataset_name]))
                if label != target_label
            ]
            # Load and concatenate all the data for non-target labels
            inputs = np.concatenate([np.load(file) for file in label_files], axis=0)
        elif attack_type == "adversarial_example":
            # For adversarial examples, run attack on a specific label to see how many are misclasified
            inputs = np.load(f"adv_datasets/{dataset_name}/{model_name}/FGSM/label_{target_label}_correct.npy")


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

        # indices = np.where(Y != target_label)[0]
        # inputs = np.array(X)[indices]

        check = model.forward(inputs)
        print(check)

        outputs = model.forward_attack(inputs, return_all_outputs, attack_type, attack_reference, fixed_point, optimised)

        return outputs