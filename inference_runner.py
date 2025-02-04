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
            # all_layer_outputs = model.forward(inputs, return_all_outputs=True, fixed_point=fixed_point)
            # avg_outputs = [np.mean(layer_output, axis=0) for layer_output in all_layer_outputs]
            avg_input = np.mean(inputs, axis=0, keepdims=True) 
            outputs_from_avg_input = model.forward(avg_input, return_all_outputs=True, fixed_point=fixed_point)
            return outputs_from_avg_input
        else:
            return model.forward(inputs, return_all_outputs=False, fixed_point=fixed_point)

    @staticmethod
    def run_inference_on_all_models(models_info, target_label, attack_type, return_all_outputs=False, fixed_point=None):
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

        if attack_type == "layer_output_matching":
            label_data_file = f"datasets/{model_name}/label_{target_label}/label_{target_label}_correct.npy"
        elif attack_type == "adversarial_example":
            label_data_file = f"adv_datasets/{dataset_name}/{model_name}/FGSM/label_{target_label}_adversarial.npy"

        print(f"Running inference for {model_name} on label {target_label} images.")
        outputs = InferenceRunner.run_inference_on_label_batch(model, label_data_file, return_all_outputs, fixed_point)

        return outputs