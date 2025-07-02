#!/usr/bin/env python3
# main.py - Entry point for TRIDENT system
import argparse
import logging
from pathlib import Path

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function that coordinates the different functionalities of the TRIDENT system
    based on the provided command-line arguments.
    """
    parser = argparse.ArgumentParser(description='TRIDENT: Tabular Representation with Transformer Encoder')
    
    # Required parameters
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name without .csv extension (e.g. vehicle_20nan)')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Training specific parameters
    training_group = parser.add_argument_group('Training parameters')
    training_group.add_argument('--label_column', type=str, default='class',
                        help='Name of the label/target column in the dataset (default: "class")')
    training_group.add_argument('--plot_losses', action='store_true',
                        help='Generate loss plots during training')
    training_group.add_argument('--save_model', action='store_true',
                        help='Save the final trained model')
    
    # Optuna specific parameters
    optuna_group = parser.add_argument_group('Optuna parameters')
    optuna_group.add_argument('--use_optuna', action='store_true',
                        help='Use Optuna for hyperparameter optimization')
    optuna_group.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials to run (default: 50)')
    optuna_group.add_argument('--retrain_best', action='store_true',
                        help='Retrain with best parameters after optimization')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.use_optuna:
        # Hyperparameter optimization mode
        logger.info(f"Starting hyperparameter optimization for dataset {args.dataset_name}")
        from opt import run_hyperparameter_optimization
        run_hyperparameter_optimization(args)
    else:
        # Regular training mode
        logger.info(f"Starting training for dataset {args.dataset_name}")
        from train import main as train_main
        train_main(args)

if __name__ == "__main__":
    main() 