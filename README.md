
# High Accuracy Neural Emulation of Boltzmann Code Predictions

## Overview
This repository presents an in-depth exploration into the impact of accuracy settings on computational speed and performance for cosmological power spectra. The focus is to evaluate **CosmoPower**, a neural network-based cosmological emulator, and analyze how different accuracy settings of training data affect its performance.

## Objective
- Assess the trade-off between accuracy and computational speed in cosmological power spectra predictions.
- Investigate CosmoPower's effectiveness across varied accuracy configurations using Boltzmann-generated training datasets.
- Provide valuable insights into optimizing emulator settings for enhanced cosmological analysis.

## Repository Structure
```
├── data/                # Datasets used for training and evaluation
├── notebooks/           # Jupyter notebooks containing analysis and visualizations
├── scripts/             # Python scripts for training and testing models
├── results/             # Outputs and plots from experiments
├── LICENSE              # Project license file
├── README.md            # Project description and instructions
└── requirements.txt     # Dependencies for replicating the environment
```

## Installation and Setup
Clone this repository and set up the necessary Python environment using:

```bash
git clone https://github.com/OriolNuez/High-Accuracy-Neural-Emulation-of-Boltzmann-Code-Predictions.git
cd High-Accuracy-Neural-Emulation-of-Boltzmann-Code-Predictions

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

## Dependencies
This project requires Python 3.8 or higher and the following core packages:

- `CosmoPower`: Neural emulator for cosmological analysis.
- `CAMB`: Code for Anisotropies in the Microwave Background, to generate cosmological power spectra.
- `Cobaya`: Bayesian analysis and sampling package.
- `TensorFlow`: Framework for training neural network models.
- `Matplotlib`: For visualization and plotting.
- `NumPy` & `SciPy`: Fundamental numerical computation libraries.

For the complete list of dependencies, see [`requirements.txt`](requirements.txt).

## Usage
After installation, you can replicate or expand upon the experiments:

1. Generate or download datasets with various accuracy settings.
2. Train CosmoPower models on these datasets.
3. Evaluate and compare models in terms of accuracy and computational speed.
4. Analyze results and visualize performance trade-offs.

Example command to run training scripts (adapt according to your specific scripts):
```bash
python scripts/train_emulator.py --config configs/high_accuracy_config.yaml
```

## Results
Detailed results, figures, and performance comparisons from the conducted experiments are located in the [`results/`](results/) folder. For detailed interpretation, see the Jupyter notebooks provided in [`notebooks/`](notebooks/).

## Contributions
Contributions and suggestions are welcome! Please open an issue to discuss ideas or submit a pull request directly to enhance the project.

## License
This project is licensed under the terms outlined in the [`LICENSE`](LICENSE) file.

## Contact
For questions or further discussions regarding this project, please contact [your email address or another preferred method of communication].
