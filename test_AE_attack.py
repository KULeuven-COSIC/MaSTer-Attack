
import numpy as np
import os
from tensorflow.keras.models import load_model
from attack_runner import AttackRunner
from inference_runner import InferenceRunner
from model_init import LABEL_RANGES

def test_attack_on_model(model_name, dataset_name, target_label, precisions=[6, 8], optimised=True):
    """
    Runs an attack on a specific model, targeting a specific label, with chosen fixed-point precisions.
    
    Parameters:
    - model_name (str): Name of the model to attack.
    - dataset_name (str): Dataset associated with the model.
    - target_label (int): The label to attack.
    - precisions (list): A list containing up to two fixed-point precision values.
    - optimised (bool): Whether to use an optimized attack method.
    """
    
    # Get the model path
    model_path = f"models/{dataset_name.lower()}/{model_name}"
    
    # Check if label range exists
    label_range = LABEL_RANGES.get(dataset_name)
    if label_range is None or target_label not in label_range:
        raise ValueError(f"Invalid label {target_label} for dataset {dataset_name}.")
    
    # Load reference matrix
    key = f"{model_name}_label_{target_label}"
    reference_matrix = InferenceRunner.run_inference_on_all_models(
        [model_name, model_path, dataset_name], target_label, "adversarial_example_PGD", return_all_outputs=True
    )

    print(len(reference_matrix))
    
    attack_results = {}
    for precision in precisions:
        print(f"Running attack on {model_name} for label {target_label} with precision {precision}...")
        attack_rate = AttackRunner.run_attack_on_all_models(
            [model_name, model_path, dataset_name],
            target_label,
            return_all_outputs=False,
            attack_type="adversarial_example_PGD",
            attack_reference=reference_matrix,
            fixed_point=precision,
            optimised=optimised
        )
        
        # Calculate success rate
        correctly_misclassified = sum(
            1 for output in attack_rate if (np.argmax(output) if len(label_range) > 2 else (1 if output >= 0.5 else 0)) != target_label
        )
        total_samples = len(attack_rate)
        success_rate = correctly_misclassified / total_samples if total_samples > 0 else 0
        attack_results[precision] = success_rate
        
    # Print results
    print("\nAttack Success Rates:")
    for precision, success_rate in attack_results.items():
        print(f"Precision {precision}: {success_rate:.2%}")
    
    return attack_results


test_attack_on_model("DNN_3_MNIST", "MNIST", target_label=8, precisions=[4])
test_attack_on_model("DNN_5_MNIST", "MNIST", target_label=8, precisions=[4])
test_attack_on_model("DNN_7_MNIST", "MNIST", target_label=8, precisions=[4])