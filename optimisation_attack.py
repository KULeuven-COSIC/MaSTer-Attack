import os
import numpy as np
from inference_runner import InferenceRunner
from attack_runner import AttackRunner
from visualiser import Visualise
from model_init import LABEL_RANGES

# Main function
def run_attack(models_info, fixed_point_precisions, optimised, realistic, budget):
    models_info = models_info

    attack_type = "optimisation_attack"
    optimised=optimised
    fixed_point_precisions = fixed_point_precisions
    budget=budget

    reference_matrices = {}
    success_rates = {precision: {} for precision in fixed_point_precisions}
    for precision in fixed_point_precisions:
        for model_name, model_path, dataset_name in models_info:
            label_range = LABEL_RANGES.get(dataset_name)
            for target_label in label_range:
                # print(f"Running inference for {model_name} on label {target_label}")
                key = f"{model_name}_label_{target_label}_precision_{precision}"
                reference_matrices[key] = InferenceRunner.optimise_reference([model_name, model_path, dataset_name], target_label, fixed_point = precision, optimised=optimised)

    attack_rate = {}
    

    # Store success rates for each precision
    success_rates = {precision: {} for precision in fixed_point_precisions}
    for precision in fixed_point_precisions:
        for model_name, model_path, dataset_name in models_info:
            label_range = LABEL_RANGES.get(dataset_name)
            for target_label in label_range:
                # print(f"Running attack on {model_name} for label {target_label}")
                ref_key = f"{model_name}_label_{target_label}_precision_{precision}"
                key = f"{model_name}_label_{target_label}"
                attack_rate[key] = AttackRunner.run_attack_on_all_models([model_name, model_path, dataset_name], target_label, return_all_outputs=False, attack_type=attack_type, attack_reference=reference_matrices[ref_key], fixed_point = precision, optimised=optimised, budget=budget, realistic=False)
    
                # Determine the number of classes for interpretation
                num_classes = len(label_range)
                
                # Process attack results
                correctly_misclassified = 0
                total_samples = len(attack_rate[key])
                
                for output in attack_rate[key]:
                    if num_classes > 2:  # Multi-class classification
                        predicted_label = np.argmax(output)
                    else:  # Binary classification
                        predicted_label = 1 if output >= 0.5 else 0
                    
                    # Check if misclassification to the target label occurred
                    if predicted_label == target_label:
                        correctly_misclassified += 1
                
                # Calculate success rate
                success_rate = correctly_misclassified / total_samples if total_samples > 0 else 0
                
                # Store the success rate
                success_rates[precision][key] = success_rate

    # print(attack_rate[f"DNN_3_MNIST_label_1"][:10])
    # print(attack_rate[f"DNN_5_MITBIH_label_2"][:10])
    # print(attack_rate[f"DNN_5_VOICE_label_0"][:10])
    # print(attack_rate[f"DNN_5_VOICE_label_1"][:10])
    # print(attack_rate[f"DNN_5_OBESITY_label_6"][:10])

    # Print success rates for verification
    for precision, rates in success_rates.items():
        print(f"Fixed-Point Precision: {precision}")
        for key, rate in rates.items():
            print(f"  {key}: Success Rate = {rate:.2%}")

    Visualise.plot_success_rate_by_labels_and_datasets(success_rates, save_dir='label_plots/opt_attack')
    Visualise.plot_success_rate_by_models(success_rates, save_dir="model_plots/opt_attack")