import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from attack_runner import AttackRunner
from inference_runner import InferenceRunner
from visualiser import Visualise
from model_init import ModelInitializer, LABEL_RANGES

DEBUG = False

def evaluate_and_group_by_label(model, x_data, y_data, attack_fn, attack_params, batch_size=128):
    """
    Evaluates the model with an attack, groups correctly classified samples by label,
    and extracts the corresponding adversarial examples.

    Returns:
    - correctly_classified: Dict of correctly classified samples by label.
    - adversarial_examples: Dict of adversarial examples corresponding to correctly classified samples.
    - label_accuracies: Dictionary mapping labels to accuracy on adversarial samples.
    """
    # Handle both integer and one-hot encoded labels
    if len(y_data.shape) > 1 and y_data.shape[1] > 1:
        true_labels = np.argmax(y_data.numpy(), axis=1)  # Convert one-hot to integers
    else:
        true_labels = y_data.numpy()  # Already integer labels

    # Get total number of samples per label
    unique_labels, label_counts = np.unique(true_labels, return_counts=True)
    total_samples_per_label = dict(zip(unique_labels, label_counts))

    # Storage for correctly classified samples & adversarial examples
    correctly_classified = {label: [] for label in unique_labels}
    adversarial_examples = {label: [] for label in unique_labels}
    label_accuracies = {label: [] for label in unique_labels}

    num_samples = x_data.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))

    for batch_index in range(num_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, num_samples)

        x_batch = tf.cast(x_data[start:end], tf.float32)
        y_batch = true_labels[start:end]

        # Get predictions on clean data
        clean_preds = np.argmax(model.predict(x_batch, verbose=0), axis=1)

        # Identify correctly classified samples
        correctly_mask = clean_preds == y_batch 
        correct_samples = x_batch.numpy()[correctly_mask]
        correct_labels = y_batch[correctly_mask]

        if correct_samples.shape[0] == 0:
            raise Exception

        # Generate adversarial examples only for correctly classified samples
        adv_examples = attack_fn(model, tf.convert_to_tensor(correct_samples), **attack_params).numpy()
        adv_preds = np.argmax(model.predict(adv_examples, verbose=0), axis=1)

        # Store samples by label
        for i, label in enumerate(correct_labels):
            correctly_classified[label].append(correct_samples[i])
            adversarial_examples[label].append(adv_examples[i])

    # Convert lists to NumPy arrays
    for label in correctly_classified.keys():
        correctly_classified[label] = np.array(correctly_classified[label])
        adversarial_examples[label] = np.array(adversarial_examples[label])

        total_label_samples = total_samples_per_label[label]  # Total samples for this label in dataset
        correctly_classified_count = len(correctly_classified[label])

        if correctly_classified_count > 0:
            adv_preds = np.argmax(model.predict(adversarial_examples[label], verbose=0), axis=1)
            adv_correct = np.sum(adv_preds == label)

            clean_acc = correctly_classified_count / total_label_samples  # Accuracy per label
            adv_acc = adv_correct / correctly_classified_count  # Accuracy after attack
            attack_success = 1 - adv_acc  # Attack effectiveness

            label_accuracies[label] = (clean_acc, adv_acc, attack_success)
        else:
            label_accuracies[label] = (0.0, 0.0, 0.0)  # No correctly classified samples for this label

    return correctly_classified, adversarial_examples, label_accuracies


def save_grouped_samples(dataset_name, model_name, attack_type, correctly_classified, adversarial_examples, label_accuracies, output_dir="adv_datasets"):
    """
    Saves the correctly classified samples and their corresponding adversarial examples,
    organizing them by dataset and model.

    Parameters:
    - dataset_name: Name of the dataset used.
    - model_name: Name of the model evaluated.
    - correctly_classified: Dictionary {label: correctly classified samples}
    - adversarial_examples: Dictionary {label: adversarial examples}
    - output_dir: Root directory where output files are saved.
    """
    model_output_dir = os.path.join(output_dir, dataset_name, model_name, attack_type)
    os.makedirs(model_output_dir, exist_ok=True)

    for label in correctly_classified.keys():
        correct_path = os.path.join(model_output_dir, f"label_{label}_correct.npy")
        adversarial_path = os.path.join(model_output_dir, f"label_{label}_adversarial.npy")

        np.save(correct_path, np.array(correctly_classified[label]))
        np.save(adversarial_path, np.array(adversarial_examples[label]))

        clean_acc, adv_acc, attack_success = label_accuracies[label]
        if DEBUG:
            print(f"Label {label}:")
            print(f" - Clean Accuracy: {clean_acc * 100:.2f}%")
            print(f" - Adversarial Accuracy: {adv_acc * 100:.2f}%")
            print(f" - Attack Success Rate: {attack_success * 100:.2f}%\n")

    print("\nData saving completed!")


def load_and_evaluate_models(dataset_configs, base_model_dir="models"):
    results = []  # To store evaluation results for all models

    for dataset_config in dataset_configs:
        dataset_name = dataset_config["dataset_name"]
        data_loader = dataset_config["data_loader"]
        models_info = dataset_config["models_info"]

        print(f"\n--- Processing Dataset: {dataset_name} ---")

        # Load dataset
        data = data_loader()
        if len(data) == 4:
            x_train, y_train, x_test, y_test = data
            x_train, x_test = tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_test)
            y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)
        elif len(data) == 2:
            x_train, y_train = data
            x_test, y_test = None, None
            x_train = tf.convert_to_tensor(x_train)
            y_train = tf.convert_to_tensor(y_train)

        for model_name, _ in models_info:
            print(f"\n--- Loading Model: {model_name} ---")

            # Load the pre-trained model
            model_path = os.path.join(base_model_dir, dataset_name.lower(), model_name, f"{model_name}.h5")
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}. Skipping...")
                continue

            model = load_model(model_path)
            print(f"Loaded model from {model_path}.")

            # Evaluate on clean data
            if x_test is not None:
                clean_loss, clean_acc = model.evaluate(x_test, y_test, verbose=0)
                print(f"Clean Accuracy for {model_name}: {clean_acc * 100:.2f}%")
            else:
                print(f"No test set available for {dataset_name}. Skipping.")

            # Generate adversarial examples and evaluate
            if x_test is not None:
                eps = 0.3  # Default perturbation size
                step_size = 0.01  # Step size for PGD
                num_steps = 40  # Iterations for PGD

                # FGSM evaluation
                fgsm_params = {"eps": eps, "norm": np.inf}
                fgsm_correct, fgsm_adversarial, pgd_label_accuracies = evaluate_and_group_by_label(
                    model, x_test, y_test, fast_gradient_method, fgsm_params, batch_size=128
                )
                save_grouped_samples(dataset_name, model_name, "FGSM", fgsm_correct, fgsm_adversarial, pgd_label_accuracies)

                # PGD evaluation
                pgd_params = {"eps": eps, "eps_iter": step_size, "nb_iter": num_steps, "norm": np.inf}
                pgd_correct, pgd_adversarial, pgd_label_accuracies = evaluate_and_group_by_label(
                    model, x_test, y_test, projected_gradient_descent, pgd_params, batch_size=128
                )
                save_grouped_samples(dataset_name, model_name, "PGD", pgd_correct, pgd_adversarial, pgd_label_accuracies)

    return results


dataset_configs = ModelInitializer.get_tf_config()

results = load_and_evaluate_models(dataset_configs)

# Print summary
for result in results:
    print(result)


models_info = [
        # ("DNN_3_MNIST", "models/mnist/DNN_3_MNIST", "MNIST"),
        # ("DNN_5_MNIST", "models/mnist/DNN_5_MNIST", "MNIST"),
        # ("DNN_7_MNIST", "models/mnist/DNN_7_MNIST", "MNIST"),
        # ("DNN_3_CIFAR10", "models/cifar10/DNN_3_CIFAR10", "CIFAR10"),
        ("DNN_5_CIFAR10", "models/cifar10/DNN_5_CIFAR10", "CIFAR10"),
        # ("DNN_3_MITBIH", "models/mitbih/DNN_3_MITBIH", "MITBIH"),
        # ("DNN_5_MITBIH", "models/mitbih/DNN_5_MITBIH", "MITBIH")
        # ("DNN_5_VOICE", "models/voice/DNN_5_VOICE", "VOICE"),
        # ("DNN_5_OBESITY", "models/obesity/DNN_5_OBESITY", "OBESITY")
    ]

attack_type = "adversarial_example_PGD"

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

reference_matrices = {}
for model_name, model_path, dataset_name in models_info:
    label_range = LABEL_RANGES.get(dataset_name)
    for target_label in label_range:
        for technique in techniques:
            # print(f"Running inference for {model_name} on label {target_label}")
            key = f"{model_name}_label_{target_label}_tech_{technique}"
            reference_matrices[key] = InferenceRunner.run_inference_on_all_models([model_name, model_path, dataset_name], target_label, attack_type, return_all_outputs=True, technique=technique)

# mnist_model = "DNN_5_MNIST"
# mitbih_model = "DNN_5_MITBIH"

attack_rate = {}
optimised=False
budget=False
realistic=True

# Define the fixed-point precisions to test
fixed_point_precisions = [8, 9, 10, 11, 12, 13, 14, 15, 16]

# Store success rates for each precision
success_rates = {precision: {} for precision in fixed_point_precisions}
for precision in fixed_point_precisions:
    for model_name, model_path, dataset_name in models_info:
        label_range = LABEL_RANGES.get(dataset_name)
        for target_label in label_range:
            for technique in techniques:
                # print(f"Running attack on {model_name} for label {target_label}")
                key = f"{model_name}_label_{target_label}_tech_{technique}"
                attack_rate[key] = AttackRunner.run_attack_on_all_models([model_name, model_path, dataset_name], target_label, return_all_outputs=False, attack_type=attack_type, attack_reference=reference_matrices[key], fixed_point = precision, optimised=optimised, budget=budget, realistic=realistic)

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
                    if predicted_label != target_label:
                        correctly_misclassified += 1
                
                # Calculate success rate
                success_rate = correctly_misclassified / total_samples if total_samples > 0 else 0
                
                # Store the success rate
                success_rates[precision][key] = success_rate

# Print success rates for verification
for precision, rates in success_rates.items():
    print(f"Fixed-Point Precision: {precision}")
    for key, rate in rates.items():
        print(f"  {key}: Success Rate = {rate:.2%}")

Visualise.plot_success_rate_by_labels(success_rates, save_dir=f'label_plots/{attack_type}')
Visualise.plot_success_rate_by_models(success_rates, save_dir=f'model_plots/{attack_type}')
Visualise.plot_success_rate_by_models_techniques(success_rates, save_dir='technique_plots/ae_attack')