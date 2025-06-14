import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

def parse_key(key):
    """
    Parses a key in one of the following formats:
    - "modelName_label_targetLabel_tech_technique"
    - "modelName_label_targetLabel"
    Returns:
    model_name, target_label, technique (or None if not present)
    """
    if "_tech_" in key:
        model_name, remainder = key.split("_label_", 1)
        target_label, technique = remainder.split("_tech_", 1)
    else:
        model_name, target_label = key.split("_label_", 1)
        technique = None
    return model_name, target_label, technique

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
            filename_tikz = os.path.join(save_dir, f"{model}_success_rate.tex")
            tikzplotlib.save(filename_tikz)
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
            filename_tikz = os.path.join(save_dir, f"{dataset_name}_label_{label}_success_rate.tex")
            tikzplotlib.save(filename_tikz)
            plt.savefig(filename)
            plt.close()


    def plot_success_rate_by_labels_and_datasets(success_rates, save_dir="plots"):
        """
        Generates two sets of plots:
        1. Per dataset-label plots: For each (dataset, label) pair, plot success rates of different models.
        2. Average plots: For each dataset, plot the average success rate across labels per model.

        The function supports keys in two formats:
        - With technique: "modelName_label_targetLabel_tech_technique"
        - Without technique: "modelName_label_targetLabel"
        
        Args:
            success_rates (dict): Structure {precision: {"key": success_rate}}
            save_dir (str): Base directory for saving plots.
        """
        # Create directories to store plots
        label_plots_dir = os.path.join(save_dir, "by_label")
        avg_plots_dir = os.path.join(save_dir, "average")
        os.makedirs(label_plots_dir, exist_ok=True)
        os.makedirs(avg_plots_dir, exist_ok=True)
        
        # # -------------------------------------
        # # 1. Create per dataset-label plots.
        # # -------------------------------------
        # # Group keys by (dataset, target_label)
        # dataset_label_to_models = {}  # {(dataset, target_label): set(model_name)}
        # for precision in success_rates:
        #     for key in success_rates[precision]:
        #         model_name, target_label, _ = parse_key(key)
        #         # Assume dataset is the last part of the model name (e.g., "DNN_3_MNIST" -> "MNIST")
        #         dataset = model_name.split("_")[-1]
        #         dataset_label_to_models.setdefault((dataset, target_label), set()).add(model_name)
        
        # # Plot success rate for each (dataset, label) pair.
        # for (dataset, target_label), models in dataset_label_to_models.items():
        #     fig, ax = plt.subplots(figsize=(8, 6))
        #     precisions = sorted(success_rates.keys())
        #     for model in models:
        #         y_values = []
        #         for precision in precisions:
        #             # Look for a key that matches the current model and target label (ignoring technique if present)
        #             found_value = None
        #             for key, value in success_rates[precision].items():
        #                 m, t, _ = parse_key(key)
        #                 if m == model and t == target_label:
        #                     found_value = value * 100  # Convert to percentage
        #                     break
        #             y_values.append(found_value if found_value is not None else 0)
        #         ax.plot(precisions, y_values, marker="o", label=model)
            
        #     ax.set_title(f"Success Rate for {dataset.upper()} - Label {target_label}", fontsize=14)
        #     ax.set_xlabel("Fixed-Point Precision", fontsize=12)
        #     ax.set_ylabel("Success Rate (%)", fontsize=12)
        #     ax.grid(True, linestyle="--", alpha=0.6)
        #     ax.legend(fontsize=10)
            
        #     png_filename = os.path.join(label_plots_dir, f"{dataset}_label_{target_label}_success_rate.png")
        #     tikz_filename = os.path.join(label_plots_dir, f"{dataset}_label_{target_label}_success_rate.tex")
        #     plt.savefig(png_filename)
        #     try:
        #         tikzplotlib.save(tikz_filename)
        #     except Exception as e:
        #         print(f"Could not save tikz file: {e}")
        #     plt.close()
        
        # -------------------------------------------
        # 2. Create average success rate per dataset.
        # -------------------------------------------
        # We need to average across labels for each model in each dataset.
        # Structure: {dataset: {model: {precision: [list of success rates]}}}
        dataset_model_rates = {}
        for precision, rates in success_rates.items():
            for key, rate in rates.items():
                model_name, target_label, _ = parse_key(key)
                dataset = model_name.split("_")[-1]
                dataset_model_rates.setdefault(dataset, {}).setdefault(model_name, {}).setdefault(precision, []).append(rate)
        
        # Now compute the average success rate for each model and precision.
        dataset_model_avg = {}
        for dataset, models in dataset_model_rates.items():
            dataset_model_avg.setdefault(dataset, {})
            for model, prec_dict in models.items():
                dataset_model_avg[dataset].setdefault(model, {})
                for precision, rate_list in prec_dict.items():
                    avg_rate = np.mean(rate_list)
                    dataset_model_avg[dataset][model][precision] = avg_rate
        
        # Create an average plot for each dataset.
        for dataset, models_data in dataset_model_avg.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            precisions = sorted(success_rates.keys())
            for model, prec_data in models_data.items():
                y_values = [prec_data.get(precision, 0) * 100 for precision in precisions]
                ax.plot(precisions, y_values, marker="o", label=model)
            
            ax.set_title(f"Average Success Rate for {dataset.upper()}", fontsize=14)
            ax.set_xlabel("Fixed-Point Precision", fontsize=12)
            ax.set_ylabel("Average Success Rate (%)", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=10)
            
            png_filename = os.path.join(avg_plots_dir, f"{dataset}_average_success_rate.png")
            tikz_filename = os.path.join(avg_plots_dir, f"{dataset}_average_success_rate.tex")
            plt.savefig(png_filename)
            try:
                tikzplotlib.save(tikz_filename)
            except Exception as e:
                print(f"Could not save tikz file: {e}")
            plt.close()

    @staticmethod
    def plot_success_rate_by_models_techniques(success_rates, save_dir="model_plots/techniques"):
        """
        New plot: For each model, plots success rates across different representative techniques.
        The x-axis represents the fixed-point precision and each line corresponds to a technique.
        Success rates are averaged over all target labels.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Organize data into: {model: {technique: {precision: [rates across labels]}}}
        model_data = {}
        for precision, rates in success_rates.items():
            for key, rate in rates.items():
                # key format: "modelName_label_targetLabel_tech_technique"
                parts = key.split("_label_")
                model_name = parts[0]
                rest = parts[1]  # e.g. "3_tech_avg_input"
                label, tech = rest.split("_tech_")
                technique = tech
                model_data.setdefault(model_name, {}).setdefault(technique, {}).setdefault(precision, []).append(rate)
    
        # Plot one graph per model.
        for model, tech_dict in model_data.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            precisions = sorted(success_rates.keys())
            for technique, prec_dict in tech_dict.items():
                avg_rates = []
                for p in precisions:
                    rates_list = prec_dict.get(p, [])
                    avg_rate = sum(rates_list) / len(rates_list) if rates_list else 0
                    avg_rates.append(avg_rate * 100)  # convert to percentage
                ax.plot(precisions, avg_rates, marker="o", label=technique)
    
            ax.set_title(f"Success Rate for Model {model} Across Techniques", fontsize=14)
            ax.set_xlabel("Fixed-Point Precision", fontsize=12)
            ax.set_ylabel("Success Rate (%)", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=10)
    
            filename = os.path.join(save_dir, f"{model}_techniques_success_rate.png")
            filename_tikz = os.path.join(save_dir, f"{model}_techniques_success_rate.tex")
            tikzplotlib.save(filename_tikz)
            plt.savefig(filename)
            plt.close()

    @staticmethod
    def plot_success_rate_by_models_techniques(success_rates, save_dir="model_plots/techniques"):
        """
        For each model, plots success rates across different representative techniques.
        The x-axis represents the fixed-point precision and each line corresponds to a technique.
        Success rates are averaged over all target labels.
        """
        os.makedirs(save_dir, exist_ok=True)
        model_data = {}
        for precision, rates in success_rates.items():
            for key, rate in rates.items():
                parts = key.split("_label_")
                model_name = parts[0]
                rest = parts[1]  # e.g. "3_tech_avg_input"
                label, tech = rest.split("_tech_")
                technique = tech
                model_data.setdefault(model_name, {}).setdefault(technique, {}).setdefault(precision, []).append(rate)
    
        for model, tech_dict in model_data.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            precisions = sorted(success_rates.keys())
            for technique, prec_dict in tech_dict.items():
                avg_rates = []
                for p in precisions:
                    rates_list = prec_dict.get(p, [])
                    avg_rate = sum(rates_list) / len(rates_list) if rates_list else 0
                    avg_rates.append(avg_rate * 100)
                ax.plot(precisions, avg_rates, marker="o", label=technique)
    
            ax.set_title(f"Success Rate for Model {model} Across Techniques", fontsize=14)
            ax.set_xlabel("Fixed-Point Precision", fontsize=12)
            ax.set_ylabel("Success Rate (%)", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=10)
    
            filename = os.path.join(save_dir, f"{model}_techniques_success_rate.png")
            filename_tikz = os.path.join(save_dir, f"{model}_techniques_success_rate.tex")
            tikzplotlib.save(filename_tikz)
            plt.savefig(filename)
            plt.close()

    @staticmethod
    def plot_fixed_point_distribution(aggregated_truncated, save_dir="analysis_plots"):
        """
        Plots the distribution of fixed-point multiplication results and the truncated bits.
        Two histograms are created side-by-side: one for the raw fixed-point multiplication results
        and one for the truncated bits (the last fixed_point bits before truncation).

        Args:
            aggregated_mult (np.ndarray or list of np.ndarray): Fixed-point multiplication results.
            aggregated_truncated (np.ndarray or list of np.ndarray): Truncated bits results.
            save_dir (str): Directory where the plot files will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # If the inputs are lists, concatenate them into single arrays.
        if isinstance(aggregated_truncated, list):
            aggregated_truncated = np.concatenate(aggregated_truncated)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Histogram for truncated bits values.
        ax.hist(aggregated_truncated, bins=50, color='salmon', edgecolor='black')
        ax.set_title("Distribution of Truncated Bits", fontsize=14)
        ax.set_xlabel("Truncated Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        filename = os.path.join(save_dir, "fixed_point_distribution.png")
        filename_tikz = os.path.join(save_dir, "fixed_point_distribution.tex")
        tikzplotlib.save(filename_tikz)
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_sign_distribution(sign_stats, save_dir="analysis_plots"):
        """
        Plots a bar chart showing the percentage of negative, zero, and positive values
        in the fixed-point multiplication results. Each bar is annotated with the raw count,
        displayed in power-of-2 notation when possible.
        
        Args:
            sign_stats (dict): Dictionary with keys:
                - 'negative', 'zero', 'positive': raw counts.
                - 'negative_pct', 'zero_pct', 'positive_pct': percentages (as fractions, e.g. 0.25 for 25%).
            save_dir (str): Directory where the plot file will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        categories = ["Negative", "Zero", "Positive"]
        # Convert fractions to percentages.
        percentages = [
            sign_stats.get("negative_pct", 0) * 100,
            sign_stats.get("zero_pct", 0) * 100,
            sign_stats.get("positive_pct", 0) * 100,
        ]
        counts = [
            sign_stats.get("negative", 0),
            sign_stats.get("zero", 0),
            sign_stats.get("positive", 0),
        ]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(categories, percentages, color=["salmon", "lightgrey", "skyblue"], edgecolor="black")
        ax.set_title("Sign Distribution", fontsize=14)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_xlabel("Sign Category", fontsize=12)
        ax.grid(True, axis='y', linestyle="--", alpha=0.6)
        
        # Annotate each bar with the raw count in power-of-2 notation.
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            annotation = Visualise.to_power_of_two(count)
            ax.annotate(annotation,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
        
        filename = os.path.join(save_dir, "sign_distribution.png")
        filename_tikz = os.path.join(save_dir, "sign_distribution.tex")
        tikzplotlib.save(filename_tikz)
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def to_power_of_two(n):
        """
        Converts an integer n to a string in the form of a power of 2 (e.g., "2^{3}")
        if n is (approximately) a power of two. Otherwise, returns the number as a string.
        """
        if n <= 0:
            return str(n)
        exponent = np.log2(n)
        if np.isclose(exponent, round(exponent), atol=1e-6):
            return f"$2^{{{int(round(exponent))}}}$"
        else:
            return str(n)