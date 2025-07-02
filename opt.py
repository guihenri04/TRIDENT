import os
import json
import copy
import pandas as pd
import numpy as np
import torch
import optuna
import argparse
from pathlib import Path
import datetime
import logging
from functools import partial
import shutil

# Import components from train.py
from train import main as train_main
from src.utils import set_global_seed

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def define_search_space(trial):
    """
    Define the hyperparameter search space for Optuna
    """
    params = {
        'DIM': trial.suggest_int('DIM', 64, 256, step=32),
        'HIDDEN_DIM': trial.suggest_int('HIDDEN_DIM', 8, 64, step=8),
        'HEADS': trial.suggest_int('HEADS', 4, 16, step=4),
        'LAYERS': trial.suggest_int('LAYERS', 1, 6, step=1),
        'DIM_FEED': trial.suggest_int('DIM_FEED', 16, 128, step=16),
        'DROPOUT': trial.suggest_float('DROPOUT', 0.1, 0.5, step=0.1),
        'EPOCHS_PRE': trial.suggest_int('EPOCHS_PRE', 20, 60, step=10),
        'BATCH': trial.suggest_categorical('BATCH', [64, 128, 256, 512]),
        'LR_PRE': trial.suggest_float('LR_PRE', 1e-5, 1e-3, log=True),
        'WEIGHT_DECAY_PRE': trial.suggest_float('WEIGHT_DECAY_PRE', 1e-5, 1e-2, log=True),
        'PROB_MASCARA': trial.suggest_float('PROB_MASCARA', 0.2, 0.6, step=0.1),
        'EPOCH_FINE': trial.suggest_int('EPOCH_FINE', 20, 60, step=10),
        'LR_FINE': trial.suggest_float('LR_FINE', 1e-5, 1e-3, log=True),
        'WEIGHT_DECAY_FINE': trial.suggest_float('WEIGHT_DECAY_FINE', 1e-5, 1e-2, log=True),
    }
    return params


class ObjectiveFunctionWrapper:
    """
    Wrapper class for the Optuna objective function to maintain state
    """
    def __init__(self, dataset_name, seed=42, output_dir=".", optuna_dir=None):
        self.dataset_name = dataset_name
        self.seed = seed
        self.output_dir = output_dir
        self.optuna_dir = optuna_dir  # Directory to store all Optuna results
        self.base_dataset_name = dataset_name.split('_')[0]
        self.best_score = 0
        self.best_params = None
        self.best_trial_number = None
        self.best_metrics = None
        
    def __call__(self, trial):
        # Define hyperparameters for this trial
        params = define_search_space(trial)
        
        # Create Args object to pass to train_main
        class Args:
            pass
        
        args = Args()
        args.dataset_name = self.dataset_name
        args.label_column = 'class'  # Default value
        
        # Use a temporary directory for trials - we will only save the best one
        temp_dir = Path(self.output_dir) / self.dataset_name / "temp_trials"
        args.output_dir = str(temp_dir)
        
        args.plot_losses = False
        args.save_model = False
        args.seed = self.seed
        args.hyperparams_override = params  # Add custom field for hyperparams
        
        # Create temporary directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        try:
            # Run the training with these hyperparameters
            metrics = train_main(args, return_metrics=True)
            
            # Get the validation and test scores
            f1_macro = metrics['f1_macro']
            
            # Report intermediate values
            trial.report(f1_macro, step=0)
            
            # Check if this is the best score so far
            if f1_macro > self.best_score:
                self.best_score = f1_macro
                self.best_params = params
                self.best_trial_number = trial.number
                self.best_metrics = metrics
                
                # Save the best parameters found so far to datasets/hiperparams
                self.save_best_params()
                
            return f1_macro
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            return 0.0  # Return worst possible score on failure
        
    def save_best_params(self):
        """
        Save the best parameters found so far to the correct location
        """
        # Create directory structure if it doesn't exist
        save_dir = Path('datasets/hiperparams') / self.base_dataset_name
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save the hyperparameters
        save_path = save_dir / f"{self.dataset_name}.json"
        
        with open(save_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
            
        logger.info(f"Saved best hyperparameters (F1={self.best_score:.4f}) to {save_path}")


def run_hyperparameter_optimization(args):
    """
    Run hyperparameter optimization with Optuna
    """
    logger.info(f"Starting hyperparameter optimization for {args.dataset_name}")
    logger.info(f"Number of trials: {args.n_trials}")
    
    # Create directory structure for Optuna results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    optuna_dir = Path(args.output_dir) / args.dataset_name / f"optuna_{timestamp}"
    optuna_dir.mkdir(exist_ok=True, parents=True)
    
    # Create the objective function
    objective = ObjectiveFunctionWrapper(
        dataset_name=args.dataset_name,
        seed=args.seed,
        output_dir=args.output_dir,
        optuna_dir=optuna_dir
    )
    
    # Create an Optuna study
    storage_name = f"sqlite:///{optuna_dir}/optuna_study.db"
    
    study = optuna.create_study(
        study_name=f"TRIDENT_{args.dataset_name}",
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
    
    # Run the optimization
    study.optimize(objective, n_trials=args.n_trials)
    
    # Report best parameters
    logger.info("\n\n" + "="*50)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best F1 macro: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*50)
    
    # Save all Optuna results in the optuna_dir
    # 1. Save best hyperparameters
    best_params_path = optuna_dir / "best_hyperparameters.json"
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Best hyperparameters saved to {best_params_path}")
    
    # 2. Also save to the standard hiperparams directory for model loading
    base_dataset_name = args.dataset_name.split('_')[0]
    hiperparams_dir = Path('datasets/hiperparams') / base_dataset_name
    hiperparams_dir.mkdir(exist_ok=True, parents=True)
    hiperparams_path = hiperparams_dir / f"{args.dataset_name}.json"
    with open(hiperparams_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Saved for model loading: {hiperparams_path}")
    
    # Retrain the model with the best parameters and save results in optuna_dir
    if args.retrain_best:
        logger.info("\nRetraining with best parameters...")
        
        # Create Args object for retraining
        class Args:
            pass
        
        final_args = Args()
        final_args.dataset_name = args.dataset_name
        final_args.label_column = 'class'  # Default value
        
        # Save the final model in the optuna directory
        final_dir = optuna_dir / "best_model"
        final_dir.mkdir(exist_ok=True)
        final_args.output_dir = str(final_dir)
        
        # Include plots when retraining
        final_args.plot_losses = True
        final_args.save_model = True
        final_args.seed = args.seed
        final_args.hyperparams_override = study.best_params
        
        # Run final training
        metrics = train_main(final_args, return_metrics=True)
        
        # Also save metrics directly in the optuna directory
        metrics_path = optuna_dir / "best_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Best model metrics saved to {metrics_path}")
        
        logger.info("Retraining completed!")
    
    # Clean up temporary directories
    temp_dir = Path(args.output_dir) / args.dataset_name / "temp_trials"
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temporary directory: {str(e)}")
    
    logger.info(f"All Optuna results saved in: {optuna_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TRIDENT: Hyperparameter optimization with Optuna')
    
    # Required parameters
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name (without the .csv extension)')
    
    # Optional parameters
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials to run (default: 50)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for Optuna results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random number generation')
    parser.add_argument('--retrain_best', action='store_true',
                        help='Retrain the model with the best parameters after optimization')
    
    args = parser.parse_args()
    run_hyperparameter_optimization(args) 