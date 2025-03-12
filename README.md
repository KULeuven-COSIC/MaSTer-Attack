# MaSTer Attack
Implementation of the techniques presented in paper "Exploring Adversarial Attacks on the MaSTer Truncation Protocol"

## Overview
Exploring adversarial influence on NN evaluation through MaSTer truncation

## Installation
### Dependencies
Ensure you have Python installed along with the necessary libraries:
```bash
pip install tensorflow numpy matplotlib seaborn h5py pandas scikit-learn argparse cleverhans tikzplotlib
```

## Usage
### Training Models
To train all models:
```bash
python3 main.py --train
```

### Running Attacks
You can choose between three attacks:
1. **Adversarial Example Attack (AE)**
2. **Inference (destination) Attack (DEST)**
3. **Optimisation Attack (OPT)**


#### Example:
To run an AE attack:
```bash
python3 main.py --attack AE --optimised --realistic --budget
```

To run a DEST attack:
```bash
python3 main.py --attack DEST --optimised --realistic --budget
```

To run an OPT attack:
```bash
python3 main.py --attack OPT --optimised --budget
```
The script runs an attack on all specified models and fixed-point precisions as specified in main.py.

## Project Structure
- `train.py` - Trains all models.
- `main.py` - Main entry point.
- `AE_attack.py` - Runs adversarial example attacks.
- `dest_attack.py` - Runs inference attacks.
- `optimisation_attack.py` - Runs optimisation attacks.
- `network.py` - Defines the neural network.
- `layers.py` - Implements layers like Dense and Conv2D.
- `data_loader.py` - Loads datasets.
- `model_init.py` - Initializes models.
- `visualiser.py` - Generates plots for analysis.

## Expected Output
Training will save models in `models/`.
Attack results will be stored in `model_plots/{attack_type}`.