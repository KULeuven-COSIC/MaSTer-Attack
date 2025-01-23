import os
import matplotlib.pyplot as plt

class Visualise:
    def plot_success_rate_by_models(success_rates, save_dir="model_plots"):
        """
        Plots success rates for each model across labels, one plot per model.
        Saves the plots in a specified directory.
        
        Args:
            success_rates (dict): Dictionary with structure:
                                {precision: {"model_label_key": success_rate}}
            save_dir (str): Directory to save the plots.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract unique models
        models = set()
        for rates in success_rates.values():
            models.update([key.rsplit("_label_", 1)[0] for key in rates.keys()])
        
        # Create plots for each model
        for model in models:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Iterate through labels and precisions to collect data for this model
            precisions = sorted(success_rates.keys())
            labels = sorted(
                set(k.rsplit("_label_", 1)[1] for k in success_rates[precisions[0]].keys() if k.startswith(model))
            )
            for label in labels:
                key = f"{model}_label_{label}"
                ax.plot(
                    precisions,
                    [success_rates[prec].get(key, 0) * 100 for prec in precisions],  # Convert rates to percentages
                    label=f"Label {label}",
                    marker="o"
                )
            
            # Customize the plot
            ax.set_title(f"Success Rate for Model {model}", fontsize=14)
            ax.set_xlabel("Fixed-Point Precision", fontsize=12)
            ax.set_ylabel("Success Rate (%)", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=10)
            
            # Save the plot in the specified directory
            filename = os.path.join(save_dir, f"{model}_success_rate.png")
            plt.savefig(filename)
            plt.close()


    def plot_success_rate_by_labels(success_rates, save_dir="label_plots"):
        """
        Plots success rates for each label, grouped by dataset.
        Each dataset-label combination will have a plot with lines for the models in that dataset.
        Saves the plots in a specified directory.
        
        Args:
            success_rates (dict): Dictionary with structure:
                                {precision: {"model_label_key": success_rate}}
            save_dir (str): Directory to save the plots.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract unique datasets and labels
        dataset_model_labels = {}
        for rates in success_rates.values():
            for key in rates.keys():
                model_name, label_part = key.rsplit("_label_", 1)
                dataset_name = model_name.split("_")[-1]  # Extract dataset name from model name
                label = label_part
                dataset_model_labels.setdefault((dataset_name, label), set()).add(model_name)
        
        # Create plots for each dataset-label combination
        for (dataset_name, label), models in dataset_model_labels.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            
            precisions = sorted(success_rates.keys())
            for model_name in models:
                key = f"{model_name}_label_{label}"
                ax.plot(
                    precisions,
                    [success_rates[prec].get(key, 0) * 100 for prec in precisions],  # Convert rates to percentages
                    label=model_name,
                    marker="o"
                )
            
            # Customize the plot
            ax.set_title(f"Success Rate for {dataset_name.upper()} - Label {label}", fontsize=14)
            ax.set_xlabel("Fixed-Point Precision", fontsize=12)
            ax.set_ylabel("Success Rate (%)", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=10)
            
            # Save the plot in the specified directory
            filename = os.path.join(save_dir, f"{dataset_name}_label_{label}_success_rate.png")
            plt.savefig(filename)
            plt.close()