import numpy as np
from data_loader import DataLoader
from model_init import ModelInitializer
from model_init import LABEL_RANGES

class AttackRunner:
    @staticmethod
    def run_attack_on_all_models(models_info, target_label, return_all_outputs, attack_type, attack_reference, fixed_point, optimised, budget, realistic):
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

        if attack_type == "layer_output_matching" or attack_type == "optimisation_attack":
            label_files = [
                f"datasets/{model_name}/label_{label}/label_{label}_correct.npy"
                for label in range(len(LABEL_RANGES[dataset_name]))
                if label != target_label
            ]
            # Load and concatenate all the data for non-target labels
            inputs = np.concatenate([np.load(file) for file in label_files], axis=0)
        elif attack_type == "adversarial_example_PGD":
            # For adversarial examples, run attack on a specific label to see how many are misclasified
            inputs = np.load(f"adv_datasets/{dataset_name}/{model_name}/PGD/label_{target_label}_correct.npy")
        elif attack_type == "adversarial_example_FGSM":
            # For adversarial examples, run attack on a specific label to see how many are misclasified
            inputs = np.load(f"adv_datasets/{dataset_name}/{model_name}/FGSM/label_{target_label}_correct.npy")

        # if model_name == 'LeNet5_CIFAR10':
        #     inputs = inputs.reshape(-1, 32, 32, 3).astype('float32') / 255

        # check = model.forward(inputs)
        # print(check[:10])

        outputs = model.forward_attack(inputs, return_all_outputs, attack_type, attack_reference, fixed_point, optimised, budget, realistic)

        return outputs