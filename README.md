# TRIDENT: Tabular Representation Inference with Dedicated Embeddings for Null Tokens

A novel Transformer-based framework for tabular data classification that employs a two-stage training paradigm: self-supervised pre-training for embedding reconstruction, followed by supervised fine-tuning. TRIDENT adapts Transformer architectures specifically for mixed tabular data containing both categorical and numerical features, with a robust focus on handling missing values

## 🎯 Overview

TRIDENT addresses the challenge of applying Transformer architectures to tabular data, especially data with high missingness, through:

1. **Pre-training Stage**:  Self-supervised learning where the model learns to reconstruct the original feature embeddings of artificially masked positions, learning the data's latent structure and inter-feature dependencies.
2. **Fine-tuning Stage**: The pre-trained model is fine-tuned end-to-end for downstream classification tasks using a classification head over the [CLS] token representation.

The framework introduces specialized components to handle tabular data heterogeneity, treating missing values as informative, learnable signals rather than noise to be imputed.


## 🏗️ Architecture

### Core Components

- **TabularEmbedder**: Converts mixed tabular data into unified embeddings
  - Categorical features: Learnable embeddings with special tokens `[MASK]` and `[NULL]`
  - Numerical features: MLP-based transformation with special token embeddings
  - Positional encoding for feature order awareness
  - CLS token for classification

- **TabularTransformerEncoder**: Multi-layer transformer encoder
  - Multi-head self-attention mechanisms
  - Layer normalization and residual connections
  - Feed-forward networks with configurable dimensions

- **TridentPretrainer**: Self-supervised pre-training model
  - Masked Language Modeling adapted for tabular data
  - Dynamic masking based on missing value density
  - MSE loss for embedding reconstruction

- **TridentClassifier**: Supervised classification model
  - Reuses pre-trained encoder representations
  - Multi-layer classification head with dropout
  - Support for class-weighted loss functions


## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TRIDENT

# Install dependencies
pip install torch pandas scikit-learn numpy matplotlib tqdm optuna
```

### Data Preparation

1. Place CSV files in `datasets/datasets_raw/`
2. Ensure target column is named `class` (or specify with `--label_column`)
3. Generate processed datasets and splits:

```bash
cd datasets
python generate_splits.py
```

This creates:
- Processed datasets with different NaN levels (20%, 40%, 60%, 80%)
- Train/validation/test splits in `processed_datasets/splits/`
- Categorical column definitions in `categorical_columns/`

### Basic Usage

```bash
# Standard training
python main.py --dataset_name vehicle_00nan

# With visualization and model saving
python main.py --dataset_name vehicle_00nan --plot_losses --save_model

# Hyperparameter optimization
python main.py --dataset_name vehicle_00nan --use_optuna --n_trials 100 --retrain_best
```

## 📋 Command Line Arguments

### Main Parameters
- `--dataset_name`: Dataset name without .csv extension (required)
- `--label_column`: Target column name (default: "class")
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Results directory (default: "results")

### Training Options
- `--plot_losses`: Generate training loss visualizations
- `--save_model`: Save trained model checkpoints

### Optuna Optimization
- `--use_optuna`: Enable hyperparameter optimization
- `--n_trials`: Number of optimization trials (default: 50)
- `--retrain_best`: Retrain with optimal parameters after search

## ⚙️ Configuration System

### Hyperparameter Management

TRIDENT supports three configuration modes:

1. **Automatic Optimization**: Using Optuna for hyperparameter search
2. **JSON Configuration**: Load from `datasets/hiperparams/{base_dataset}/{dataset_name}.json`
3. **Default Values**: Fallback configuration when no custom settings exist

### Example Configuration
```json
{
    "DIM": 128,
    "HIDDEN_DIM": 16,
    "HEADS": 16,
    "LAYERS": 2,
    "DIM_FEED": 32,
    "DROPOUT": 0.2,
    "EPOCHS_PRE": 40,
    "BATCH": 256,
    "LR_PRE": 0.00034,
    "WEIGHT_DECAY_PRE": 0.005,
    "PROB_MASCARA": 0.5,
    "EPOCH_FINE": 40,
    "LR_FINE": 0.001,
    "WEIGHT_DECAY_FINE": 0.0019
}
```

### Hyperparameter Descriptions

**Model Architecture:**
- `DIM`: Embedding dimension for features
- `HIDDEN_DIM`: Hidden dimension for numerical MLPs
- `HEADS`: Number of attention heads
- `LAYERS`: Number of transformer layers
- `DIM_FEED`: Feed-forward network dimension

**Training Configuration:**
- `EPOCHS_PRE`: Pre-training epochs
- `EPOCH_FINE`: Fine-tuning epochs
- `BATCH`: Batch size
- `DROPOUT`: Dropout probability
- `PROB_MASCARA`: Masking probability for pre-training

**Optimization:**
- `LR_PRE`: Pre-training learning rate
- `LR_FINE`: Fine-tuning learning rate
- `WEIGHT_DECAY_PRE`: Pre-training weight decay
- `WEIGHT_DECAY_FINE`: Fine-tuning weight decay

## 📁 Project Structure

```
TRIDENT/
├── main.py                 # Main entry point and argument parsing
├── train.py               # Core training logic and evaluation
├── opt.py                 # Optuna hyperparameter optimization
├── src/
│   ├── embedder.py       # TabularEmbedder implementation
│   ├── transformer.py    # TabularTransformerEncoder
│   ├── models.py         # TridentPretrainer & TridentClassifier
│   └── utils.py          # Utility functions and preprocessing
└── datasets/
    ├── datasets_raw/     # Original CSV files
    ├── processed_datasets/ # Processed data with splits
    ├── categorical_columns/ # Feature type definitions
    └── generate_splits.py # Data preprocessing pipeline
```

## 🧠 Training Methodology

### Pre-training Phase
1. **Dynamic Masking**: Randomly masks features with probability adjusted by existing null density
2. **Embedding Reconstruction**: Predicts original embeddings for masked positions
3. **Self-Supervised Learning**: No labels required, learns from data structure

### Fine-tuning Phase
1. **Pre-trained Initialization**: The model is initialized with pre-trained weights and fine-tuned end-to-end.
2. **Classification Head**: Learns task-specific predictions
3. **Class Balancing**: Optional class weights for imbalanced datasets

## 📊 Evaluation Metrics

The framework reports comprehensive classification metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Both micro and macro averaged F1
- **Precision**: Both micro and macro averaged precision
- **Recall**: Both micro and macro averaged recall
- **Confusion Matrix**: For detailed error analysis (binary tasks)

## 🎯 Example Workflows

### Basic Training
```bash
# Train on a clean dataset
python main.py --dataset_name spambase_00nan

# Train on dataset with 20% missing values
python main.py --dataset_name spambase_20nan --plot_losses
```

### Hyperparameter Optimization
```bash
# Quick optimization (50 trials)
python main.py --dataset_name credit-g_00nan --use_optuna --n_trials 50

# Thorough optimization with retraining
python main.py --dataset_name vehicle_40nan --use_optuna --n_trials 200 --retrain_best
```

### Custom Configuration
```bash
# Use custom target column
python main.py --dataset_name custom_data --label_column target

# Save results to specific directory
python main.py --dataset_name biodeg_00nan --output_dir ./experiments/biodeg
```

## 🗃️ Supported Datasets

TRIDENT has been evaluated on various tabular classification benchmarks:

- **Vehicle Classification**: Multi-class vehicle type recognition
- **Credit Risk Assessment**: Binary credit approval prediction
- **Spam Detection**: Binary email spam classification
- **Biodegradation**: Binary molecular biodegradability prediction
- **Letter Recognition**: Multi-class character recognition
- **Electrical Grid Stability**: Binary stability prediction


```bibtex
@inproceedings{rigueira2024trident,
  title={TRIDENT: Tabular Representation Inference with Dedicated Embeddings for Null Tokens},
  author={Rigueira, Pedro B. and Mello, Victoria F. and Evangelista, Guilherme H. G. and Grossi, Caio S. and Machado, Giovana A. M. and Dutenhefner, Pedro and Meira Jr., Wagner and Pappa, Gisele L.},
  booktitle={Brazilian Conference on Intelligent Systems (BRACIS)},
  year={2024}
}
```
