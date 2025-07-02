import pandas as pd
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split

GLOBAL_SEED = 42

def set_global_seed(seed: int = GLOBAL_SEED):
    """
    Sets the global seed to ensure reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def preprocess_table(data, null_token="[NULL]", p_base=0.15, fine_tunning=False):
    """
    Preprocesses the table to replace null values with the `[NULL]` token,
    applies dynamic masking with adjusted probability and ensures that:
        - At least one mask is randomly applied per row (if not in fine-tuning mode).
        - The dynamic mask doesn't overlap with null values (doesn't mask already null positions).

    Parameters
    ----------
    data : pd.DataFrame
        Original data table.
    null_token : str
        Token to replace null values (default = "[NULL]").
    p_base : float
        Base probability for dynamic masking.
    fine_tunning : bool
        Defines if it's pre-training or fine-tuning:
          - If False, we actually mask with probability p_base.
          - If True, we don't mask anything (only replace nulls with [NULL]).

    Returns
    -------
    masked_data : pd.DataFrame
        Resulting DataFrame containing `[NULL]` and `[MASK]` tokens.
    """
    # Create a boolean matrix indicating null values
    null_matrix = data.isnull()

    # Replace null values with the `[NULL]` token
    processed_data = data.mask(null_matrix, null_token)

    if not fine_tunning:
        # Calculate the proportion of null values per row
        prop_nulls = null_matrix.sum(axis=1) / data.shape[1]

        # Adjust the dynamic masking probability based on the proportion of nulls
        # Example: if a row has 30% nulls, then p_base * (1 - 0.3) => 70% of p_base
        p_dynamic = p_base * (1 - prop_nulls.values[:, None])

        # Generate a random matrix and apply the dynamic mask
        dynamic_mask = np.random.rand(*data.shape) < p_dynamic

        # Avoid masking values that are already null
        dynamic_mask[null_matrix.values] = False

        # Ensure each row has at least one masked value
        no_mask_rows = ~dynamic_mask.any(axis=1)
        for i in np.where(no_mask_rows)[0]:
            non_null_indices = np.where(~null_matrix.iloc[i].values)[0]
            if len(non_null_indices) > 0:
                random_index = np.random.choice(non_null_indices)
                dynamic_mask[i, random_index] = True

        # Apply the mask
        masked_data = processed_data.mask(dynamic_mask, "[MASK]")
        return masked_data
    else:
        # If we're in fine-tuning mode, we don't randomly mask
        # We just return with [NULL] in place of nulls
        return processed_data

def split_numeric_and_special(df, numerical_columns, device='cuda'):
    """
    Separates real numerical values from special tokens `[MASK]` and `[NULL]`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains [MASK] and [NULL] tokens in numerical columns.
    numerical_columns : list
        List of columns considered numerical.
    device : str
        'cuda' or 'cpu', depending on where the tensor will run.

    Returns
    -------
    numeric_values : torch.FloatTensor
        Tensor (n_rows, n_num_cols) with numerical values (0.0 where there are special tokens).
    mask_flags : torch.BoolTensor
        Tensor (n_rows, n_num_cols) with True where there's `[MASK]`.
    null_flags : torch.BoolTensor
        Tensor (n_rows, n_num_cols) with True where there's `[NULL]`.
    """
    numeric_values = []
    mask_flags = []
    null_flags = []
    
    for col in numerical_columns:
        col_data = df[col].values  # Can be float or string "[MASK]" or "[NULL]"
        col_numeric = []
        col_mask = []
        col_null = []
        
        for val in col_data:
            if val == "[MASK]":
                col_numeric.append(0.0)
                col_mask.append(True)
                col_null.append(False)
            elif val == "[NULL]":
                col_numeric.append(0.0)
                col_mask.append(False)
                col_null.append(True)
            else:
                col_numeric.append(float(val))
                col_mask.append(False)
                col_null.append(False)
        
        numeric_values.append(col_numeric)
        mask_flags.append(col_mask)
        null_flags.append(col_null)
    
    # Convert from lists to tensors and transpose to (n_rows, n_num_cols)
    numeric_values = torch.tensor(numeric_values, dtype=torch.float32, device=device).T
    mask_flags     = torch.tensor(mask_flags,     dtype=torch.bool,   device=device).T
    null_flags     = torch.tensor(null_flags,     dtype=torch.bool,   device=device).T
    
    return numeric_values, mask_flags, null_flags

def create_pretrain_datasets(df, categorical_columns, numerical_columns,
                            p_base=0.15, test_size=0.1, random_state=42):
    """
    Example function that:
      1) Splits train/val (e.g., 90/10)
      2) Generates masked DF (df_masked) and original DF (df_original).
      3) Returns (df_train_original, df_train_masked, df_val_original, df_val_masked).

    This allows us to perform pre-training, comparing df_masked and df_original.
    """
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    
    # Generate masked versions (with p_base=0.15)
    df_train_masked = preprocess_table(df_train.copy(), p_base=p_base, fine_tunning=False)
    df_val_masked   = preprocess_table(df_val.copy(),   p_base=p_base, fine_tunning=False)
    
    # df_train and df_val are the "original" versions
    return df_train, df_train_masked, df_val, df_val_masked 