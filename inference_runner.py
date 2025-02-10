import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from data_loader import DataLoader
from model_init import ModelInitializer
from model_init import LABEL_RANGES

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
            avg_input = np.mean(inputs, axis=0, keepdims=True) 
            # print(inputs.shape, inputs[:1, :].shape, avg_input.shape)
            # print(model.forward(avg_input, return_all_outputs=False, fixed_point=fixed_point))
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
        elif attack_type == "adversarial_example_FGSM":
            label_data_file = f"adv_datasets/{dataset_name}/{model_name}/FGSM/label_{target_label}_adversarial.npy"
        elif attack_type == "adversarial_example_PGD":
            label_data_file = f"adv_datasets/{dataset_name}/{model_name}/PGD/label_{target_label}_adversarial.npy"

        print(f"Running inference for {model_name} on label {target_label} images.")
        outputs = InferenceRunner.run_inference_on_label_batch(model, label_data_file, return_all_outputs, fixed_point)

        return outputs
    
    # @staticmethod
    # def optimize_reference(models_info, target_label, fixed_point=None, optimised=False, num_iterations=1000, learning_rate=0.01):
    #     """
    #     Optimizes reference values for all layers to maximize the target class logit.

    #     Args:
    #         models_info: Tuple containing (model_name, model_path, dataset_name).
    #         target_label: The label to optimize towards.
    #         fixed_point: Fixed point precision (if used).
    #         optimised: Whether to use an optimized version.
    #         num_iterations: Number of gradient steps.
    #         learning_rate: Learning rate for optimization.

    #     Returns:
    #         Optimized reference values for each layer and averaged perturbation.
    #     """
    #     model_name, model_path, dataset_name = models_info

    #     # Get TensorFlow model architecture from config
    #     dataset_configs = ModelInitializer.get_tf_config()
    #     model_layers = None

    #     # Find the correct model architecture
    #     for dataset in dataset_configs:
    #         if dataset["dataset_name"] == dataset_name:
    #             for model_info in dataset["models_info"]:
    #                 if model_info[0] == model_name:
    #                     model_layers = model_info[1]
    #                     break

    #     if model_layers is None:
    #         raise ValueError(f"Model {model_name} not found in TensorFlow configuration.")

    #     # Initialize model using TensorFlow API
    #     model = tf.keras.Sequential(model_layers)
    #     DataLoader.load_weights_and_biases(model, model_path)  # Load pre-trained weights

    #     # Load input data (excluding the target label)
    #     label_files = [
    #         f"datasets/{model_name}/label_{label}/label_{label}_correct.npy"
    #         for label in range(len(LABEL_RANGES[dataset_name]))
    #         if label != target_label
    #     ]
    #     inputs = np.concatenate([np.load(file) for file in label_files], axis=0)

    #     # Convert inputs to TensorFlow tensor
    #     inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

    #     # Initialize perturbation as a trainable variable
    #     a = tf.Variable(tf.zeros_like(inputs), dtype=tf.float32)

    #     # Adam optimizer
    #     optimizer = tf.keras.optimizers.Adam(learning_rate)

    #     # Optimization loop
    #     for iteration in range(num_iterations):
    #         with tf.GradientTape() as tape:
    #             # Forward pass using TensorFlow model
    #             perturbed_inputs = inputs + a
    #             logits = model(perturbed_inputs, training=False)  # Get logits directly

    #             # Define loss: maximize target class logit
    #             loss = -tf.reduce_mean(logits[:, target_label])

    #         # Compute gradients
    #         gradients = tape.gradient(loss, a)

    #         if gradients is None:
    #             print("Warning: Gradients are None!")
    #             break  # Stop if gradients are not being computed

    #         # Apply gradient updates
    #         optimizer.apply_gradients([(gradients, a)])

    #         # Apply layer-specific clamping
    #         for layer in model.layers:
    #             if fixed_point is not None and hasattr(layer, "weights") and len(layer.weights) > 0:
    #                 input_size = inputs.shape[1]
    #                 weight_size = layer.weights[0].shape[0]
    #                 limit = 1 if optimised else input_size * weight_size * (2 ** -fixed_point)
    #                 a.assign(tf.clip_by_value(a, -limit, limit))

    #         # Print loss and debug information
    #         if iteration % 100 == 0:
    #             print(f"Iteration {iteration}: Loss = {loss.numpy()}")

    #     # Compute final reference by averaging the model outputs
    #     optimized_outputs = model(inputs + a, training=False)
    #     optimized_references = tf.reduce_mean(optimized_outputs, axis=0).numpy()

    #     # Compute the averaged perturbation across samples
    #     averaged_a = tf.reduce_mean(a, axis=0).numpy()

    #     return optimized_references, averaged_a
    
    def optimise_reference(models_info, target_label, fixed_point, optimised, batch_size=128, num_epochs=10, learning_rate=0.05):
        """
        Optimizes universal layer-specific offsets (a_i) for a pre-trained model so that when these
        offsets are added to the pre-activation of each optimizable layer, the final network output
        maximizes the probability of a target class.

        For each optimizable layer (here, those with "dense" or "conv" in their name), the modified pre-activation is:
        
            pre_act = x @ kernel + bias + a
        
        where 'a' is a universal offset (with shape equal to layer.output_shape[1:]) that is the same for all samples.
        
        The allowed range for each offset is computed as:
            if optimised:
                limit = 1 * 2**(-fixed_point)
            else:
                limit = layer.input_shape[1] * 2**(-fixed_point)
        and the offsets are randomly initialized (and later clipped) to lie in [-limit, limit].

        Args:
            models_info (tuple): (model_name, model_path, dataset_name).
            target_label (int): The class index to maximize.
            fixed_point (numeric): A parameter used to compute the dynamic limit.
            optimised (bool): Determines which limit formula to use.
            batch_size (int): Mini-batch size.
            num_epochs (int): Number of epochs over the entire dataset.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            list: A list of optimized universal offset arrays (one per optimizable layer).
        """

        if fixed_point <= 9:
            num_epochs = 20
            batch_size = 64
            learning_rate = 0.005
        
        model_name, model_path, dataset_name = models_info

        # Load the pre-trained model.
        model_dir = f"models/{dataset_name.lower()}/{model_name}"
        model = tf.keras.models.load_model(os.path.join(model_dir, f"{model_name}.h5"))
        model.trainable = False  # Freeze model weights

        # Get the overall input shape (excluding batch dimension)
        input_shape = model.input_shape[1:]

        # Determine model type: for binary classification we expect a single output node.
        output_shape = model.output_shape[1:]
        is_binary_classification = (len(output_shape) == 0 or output_shape[0] == 1)

        # Load the inputs for all labels except the target label.
        # (Assumes that LABEL_RANGES[dataset_name] is defined elsewhere.)
        label_files = [
            f"datasets/{model_name}/label_{label}/label_{label}_correct.npy"
            for label in range(len(LABEL_RANGES[dataset_name])) if label != target_label
        ]
        inputs = np.concatenate([np.load(file) for file in label_files], axis=0)
        num_samples = inputs.shape[0]

        # Convert inputs to the model's dtype.
        common_dtype = model.weights[0].dtype  # assume all weights share the same dtype
        inputs = inputs.astype(common_dtype.as_numpy_dtype)

        # Shuffle the dataset initially.
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        inputs = inputs[indices]

        # Identify the optimizable layers and initialize a universal offset for each.
        # Also compute a per-layer limit based on 'optimised' and 'fixed_point'.
        optimizable_layer_flags = []  # one flag per layer in the model: True if we add an offset.
        a_layers = []               # list of universal offset tf.Variable for each optimizable layer.
        limits = []                 # list of computed limits (one per optimizable layer)
        for layer in model.layers:
            if "dense" in layer.name or "conv" in layer.name:
                optimizable_layer_flags.append(True)
                # For the limit, try to use layer.input_shape[1] (the number of features).
                try:
                    layer_input_dim = layer.input_shape[1]
                except Exception:
                    layer_input_dim = 1
                if optimised:
                    limit = 1 * (2 ** (-fixed_point))
                else:
                    limit = layer_input_dim * (2 ** (-fixed_point))
                limits.append(limit)
                # Universal offset: shape is layer.output_shape[1:]
                a_shape = tuple(layer.output_shape[1:])
                a_i = tf.Variable(tf.random.uniform(a_shape, minval=-limit, maxval=limit, dtype=common_dtype))
                a_layers.append(a_i)
            else:
                optimizable_layer_flags.append(False)

        # Create an optimizer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compute the number of batches per epoch.
        batches_per_epoch = num_samples // batch_size + (1 if num_samples % batch_size != 0 else 0)

        print(f"Starting optimisation for {model_name} for target label {target_label} on fixed_point {fixed_point}")
        # print(f'Number of samples available: {num_samples}')

        # Create a tf.data.Dataset for the inputs.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.shuffle(buffer_size=num_samples).batch(batch_size)
        
        @tf.function(reduce_retracing=True)
        def train_step(x_batch):
            # Ensure the batch tensor has the common dtype.
            x_batch = tf.cast(x_batch, common_dtype)
            with tf.GradientTape() as tape:
                x = x_batch
                offset_idx = 0
                # Forward pass.
                for layer, flag in zip(model.layers, optimizable_layer_flags):
                    if flag:
                        # For dense or conv layers: pre_act = x @ kernel + bias + universal a.
                        kernel = tf.cast(layer.kernel, common_dtype)
                        bias = tf.cast(layer.bias, common_dtype)
                        pre_act = tf.matmul(x, kernel) + bias + a_layers[offset_idx]
                        if layer.activation is not None:
                            x = layer.activation(pre_act)
                        else:
                            x = pre_act
                        offset_idx += 1
                    else:
                        x = layer(x)
                logits = x

                # Compute loss.
                if is_binary_classification:
                    probs = tf.sigmoid(logits)
                    if target_label == 1:
                        loss = -tf.reduce_mean(probs)
                    else:
                        loss = tf.reduce_mean(probs)
                else:
                    probs = tf.nn.softmax(logits, axis=1)
                    loss = -tf.reduce_mean(probs[:, target_label])

            gradients = tape.gradient(loss, a_layers)
            optimizer.apply_gradients(zip(gradients, a_layers))
            for j in range(len(a_layers)):
                a_layers[j].assign(tf.clip_by_value(a_layers[j], -limits[j], limits[j]))
            return loss
        
        # Training loop over epochs.
        for epoch in range(num_epochs):
            for x_batch in dataset:
                loss = train_step(x_batch)

            # Print summary statistics for each universal offset at the end of the epoch.
            for j, a_i in enumerate(a_layers):
                a_np = a_i.numpy()
                # print(f"Epoch {epoch} - a_layer {j}: mean = {np.mean(a_np):.6f}, std = {np.std(a_np):.6f}")

        # Print shapes of the universal offsets and return them.
        # print("Final shapes of a_layers:", [a.shape for a in a_layers])
        # if target_label == 6:
        #     print([a.numpy() for a in a_layers])
        return [a.numpy() for a in a_layers]
    

    
    def optimise_reference_with_logging(models_info, target_label, fixed_point, optimised,
                                    batch_size=128, num_epochs=10, learning_rate=0.05):
        """
        A modified version of your optimise_reference function that logs the training loss (averaged
        per epoch) and the average norm of each universal offset (averaged over layers) at the end
        of each epoch.
        
        Returns:
            loss_history: A list of average losses (one per epoch).
            offset_norm_history: A list of average offset norms (one per epoch; here averaged over all layers).
            final_offsets: The final universal offsets.
        """
        model_name, model_path, dataset_name = models_info

        # Load the pre-trained model.
        model_dir = f"models/{dataset_name.lower()}/{model_name}"
        model = tf.keras.models.load_model(os.path.join(model_dir, f"{model_name}.h5"))
        model.trainable = False  # Freeze model weights

        input_shape = model.input_shape[1:]
        output_shape = model.output_shape[1:]
        is_binary_classification = (len(output_shape) == 0 or output_shape[0] == 1)

        # Load inputs for all labels except the target label.
        label_files = [
            f"datasets/{model_name}/label_{label}/label_{label}_correct.npy"
            for label in range(len(LABEL_RANGES[dataset_name])) if label != target_label
        ]
        inputs = np.concatenate([np.load(file) for file in label_files], axis=0)
        num_samples = inputs.shape[0]

        # Set a common dtype from the model’s weights.
        common_dtype = model.weights[0].dtype
        inputs = inputs.astype(common_dtype.as_numpy_dtype)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        inputs = inputs[indices]

        # Initialize universal offsets and compute per-layer limits.
        optimizable_layer_flags = []
        a_layers = []
        limits = []
        for layer in model.layers:
            if "dense" in layer.name or "conv" in layer.name:
                optimizable_layer_flags.append(True)
                try:
                    layer_input_dim = layer.input_shape[1]
                except Exception:
                    layer_input_dim = 1
                if optimised:
                    limit = 1 * (2 ** (-fixed_point))
                else:
                    limit = layer_input_dim * (2 ** (-fixed_point))
                limits.append(limit)
                a_shape = tuple(layer.output_shape[1:])
                a_i = tf.Variable(tf.random.uniform(a_shape, minval=-limit, maxval=limit, dtype=common_dtype))
                a_layers.append(a_i)
            else:
                optimizable_layer_flags.append(False)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        batches_per_epoch = num_samples // batch_size + (1 if num_samples % batch_size != 0 else 0)

        # Use tf.data for batching.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.shuffle(buffer_size=num_samples).batch(batch_size)

        # Optionally, define the training step as a tf.function.
        @tf.function(reduce_retracing=True)
        def train_step(x_batch):
            x_batch = tf.cast(x_batch, common_dtype)
            with tf.GradientTape() as tape:
                x = x_batch
                offset_idx = 0
                for layer, flag in zip(model.layers, optimizable_layer_flags):
                    if flag:
                        kernel = tf.cast(layer.kernel, common_dtype)
                        bias = tf.cast(layer.bias, common_dtype)
                        pre_act = tf.matmul(x, kernel) + bias + a_layers[offset_idx]
                        x = layer.activation(pre_act) if layer.activation is not None else pre_act
                        offset_idx += 1
                    else:
                        x = layer(x)
                logits = x
                if is_binary_classification:
                    probs = tf.sigmoid(logits)
                    loss = -tf.reduce_mean(probs) if target_label == 1 else tf.reduce_mean(probs)
                else:
                    probs = tf.nn.softmax(logits, axis=1)
                    loss = -tf.reduce_mean(probs[:, target_label])
            gradients = tape.gradient(loss, a_layers)
            optimizer.apply_gradients(zip(gradients, a_layers))
            for j in range(len(a_layers)):
                a_layers[j].assign(tf.clip_by_value(a_layers[j], -limits[j], limits[j]))
            return loss

        loss_history = []
        offset_norm_history = []  # We will record the average norm across all layers at each epoch.
        for epoch in range(num_epochs):
            epoch_losses = []
            for x_batch in dataset:
                loss = train_step(x_batch)
                epoch_losses.append(loss.numpy())
            avg_loss = np.mean(epoch_losses)
            loss_history.append(avg_loss)
            # Compute average norm for this epoch (averaging over all layers).
            norms = [np.linalg.norm(a.numpy()) for a in a_layers]
            avg_norm = np.mean(norms)
            offset_norm_history.append(avg_norm)
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}, Avg offset norm = {avg_norm:.6f}")
        
        final_offsets = [a.numpy() for a in a_layers]
        return loss_history, offset_norm_history, final_offsets