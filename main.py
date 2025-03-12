import argparse
import os
from train import train_all_models
import dest_attack
import AE_attack
import optimisation_attack

def main():
    parser = argparse.ArgumentParser(description="Run training and attacks for the project.")
    parser.add_argument("--train", action="store_true", help="Train all models.")
    parser.add_argument("--attack", choices=["AE", "DEST", "OPT"], help="Run a specific attack: AE (Adversarial Examples), DEST (Destruction), or OPT (Optimisation).")
    parser.add_argument("--optimised", action="store_true", help="Run attack on optimised implementation of truncation.")
    parser.add_argument("--realistic", action="store_true", help="Run realistic attack with general attack vector.")
    parser.add_argument("--budget", action="store_true", help="Use a limited budget for the attacker.")
    
    args = parser.parse_args()
    
    if args.train:
        print("Training all models...")
        train_all_models()
        print("Training complete.")
    elif args.attack:
        
        models_info = [
            # ("DNN_3_MNIST", "models/mnist/DNN_3_MNIST", "MNIST"),
            ("DNN_5_MNIST", "models/mnist/DNN_5_MNIST", "MNIST"),
            # ("DNN_7_MNIST", "models/mnist/DNN_7_MNIST", "MNIST"),
            # ("DNN_9_MNIST", "models/mnist/DNN_9_MNIST", "MNIST"),
            # ("DNN_3_CIFAR10", "models/cifar10/DNN_3_CIFAR10", "CIFAR10"),
            # ("DNN_5_CIFAR10", "models/cifar10/DNN_5_CIFAR10", "CIFAR10"),
            # ("LeNet5_CIFAR10", "models/cifar10/LeNet5_CIFAR10", "CIFAR10"),
            # ("DNN_3_MITBIH", "models/mitbih/DNN_3_MITBIH", "MITBIH"),
            # ("DNN_5_MITBIH", "models/mitbih/DNN_5_MITBIH", "MITBIH"),
            # ("DNN_5_VOICE", "models/voice/DNN_5_VOICE", "VOICE"),
            # ("DNN_5_OBESITY", "models/obesity/DNN_5_OBESITY", "OBESITY"),
        ]

        fixed_point_precisions = [8, 9, 10, 11, 12, 13, 14, 15, 16]
        
        if args.attack == "AE":
            print(f"Running Adversarial Example attack on {models_info}...")
            AE_attack.run_attack(models_info, fixed_point_precisions, args.optimised, args.realistic, args.budget)
        elif args.attack == "DEST":
            print(f"Running Destruction attack on {models_info}...")
            dest_attack.run_attack(models_info, fixed_point_precisions, args.optimised, args.realistic, args.budget)
        elif args.attack == "OPT":
            print(f"Running Optimisation attack on {models_info}...")
            optimisation_attack.run_attack(models_info, fixed_point_precisions, args.optimised, args.realistic, args.budget)
        
        print("Attack complete.")
    
    else:
        print("No valid option selected. Run with --help for usage details.")

if __name__ == "__main__":
    main()
