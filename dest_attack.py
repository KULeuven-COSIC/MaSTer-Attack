import os
import numpy as np
from inference_runner import InferenceRunner
from attack_runner import AttackRunner
from visualiser import Visualise
from model_init import LABEL_RANGES

if __name__ == '__main__':
    # List of models to test.
    models_info = [
        # ("DNN_3_MNIST", "models/mnist/DNN_3_MNIST", "MNIST"),
        # ("DNN_5_MNIST", "models/mnist/DNN_5_MNIST", "MNIST"),
        # ("DNN_3_CIFAR10", "models/cifar10/DNN_3_CIFAR10", "CIFAR10"),
        ("DNN_5_CIFAR10", "models/cifar10/DNN_5_CIFAR10", "CIFAR10"),
        # ("LeNet5_CIFAR10", "models/cifar10/LeNet5_CIFAR10", "CIFAR10"),
        # ("DNN_3_MITBIH", "models/mitbih/DNN_3_MITBIH", "MITBIH"),
        # ("DNN_5_MITBIH", "models/mitbih/DNN_5_MITBIH", "MITBIH"),
        # ("DNN_5_VOICE", "models/voice/DNN_5_VOICE", "VOICE"),
        # ("DNN_5_OBESITY", "models/obesity/DNN_5_OBESITY", "OBESITY"),
    ]
    
    attack_type = "layer_output_matching"
    
    # List of representative techniques.
    techniques = [
        'avg_input',
        'median_input',
        'medoid_input',         # medoid in input space
        'avg_output',
        'median_output',
        'medoid_output',   # medoid in output space
        'closest_one_hot',
        'random_sample'
    ]
    
    # First, build reference matrices for each (model, target_label, technique).
    # We assume that InferenceRunner.run_inference_on_all_models now accepts a `technique` argument.
    reference_matrices = {}
    for model_name, model_path, dataset_name in models_info:
        label_range = LABEL_RANGES.get(dataset_name)
        for target_label in label_range:
            for technique in techniques:
                key = f"{model_name}_label_{target_label}_tech_{technique}"
                reference_matrices[key] = InferenceRunner.run_inference_on_all_models(
                    [model_name, model_path, dataset_name],
                    target_label,
                    attack_type,
                    return_all_outputs=True,
                    technique=technique
                )
    
    # Prepare to run the attack for various fixed-point precisions.
    optimised = False
    budget = False
    realistic = True
    fixed_point_precisions = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    # Structure: success_rates[precision][key] where key is:
    # "modelName_label_targetLabel_tech_technique"
    success_rates = {precision: {} for precision in fixed_point_precisions}
    
    for precision in fixed_point_precisions:
        for model_name, model_path, dataset_name in models_info:
            label_range = LABEL_RANGES.get(dataset_name)
            for target_label in label_range:
                for technique in techniques:
                    key = f"{model_name}_label_{target_label}_tech_{technique}"
                    # AttackRunner.run_attack_on_all_models is assumed to accept a 'technique' argument.
                    attack_results = AttackRunner.run_attack_on_all_models(
                        [model_name, model_path, dataset_name],
                        target_label,
                        return_all_outputs=False,
                        attack_type=attack_type,
                        attack_reference=reference_matrices[key],
                        fixed_point=precision,
                        optimised=optimised,
                        budget=budget,
                        realistic=realistic,
                    )
    
                    # Determine number of classes for interpretation.
                    num_classes = len(LABEL_RANGES.get(dataset_name))
                    correctly_misclassified = 0
                    total_samples = len(attack_results)
    
                    for output in attack_results:
                        if num_classes > 2:  # Multi-class classification
                            predicted_label = np.argmax(output)
                        else:  # Binary classification
                            predicted_label = 1 if output >= 0.5 else 0
                        # Since every sample in the batch is classified to the target,
                        # a successful attack is when the prediction equals target_label.
                        if predicted_label == target_label:
                            correctly_misclassified += 1
    
                    success_rate = correctly_misclassified / total_samples if total_samples > 0 else 0
                    success_rates[precision][key] = success_rate

    # Print success rates for verification
    for precision, rates in success_rates.items():
        print(f"Fixed-Point Precision: {precision}")
        for key, rate in rates.items():
            print(f"  {key}: Success Rate = {rate:.2%}")

    Visualise.plot_success_rate_by_labels(success_rates, save_dir='label_plots/dest_attack')
    Visualise.plot_success_rate_by_models(success_rates, save_dir="model_plots/dest_attack")
    Visualise.plot_success_rate_by_models_techniques(success_rates, save_dir='technique_plots/dest_attack')