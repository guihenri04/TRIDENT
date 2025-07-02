import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import re

from .utils import split_numeric_and_special

class TabularEmbedder(nn.Module):
    """
    Class that encapsulates the creation of embeddings for tabular data:
      - Categorical columns: Uses nn.Embedding + LabelEncoder (with [MASK]/[NULL])
      - Numerical columns: MLP + special embeddings for [MASK] and [NULL]
      - [CLS] token + Positional Embedding
    """
    def __init__(self, df, categorical_columns, numerical_columns,
                 dimensao=4, hidden_dim=32):
        super().__init__()
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.dimensao = dimensao
        self.hidden_dim = hidden_dim
        
        # Store original column names and their sanitized versions for ModuleDict keys
        self.column_to_key = {}
        for col in self.numerical_columns:
            # Replace invalid characters with underscore for module names
            self.column_to_key[col] = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        
        # -----------------------
        # 1) LabelEncoders
        #    (including [MASK] and [NULL] in the vocabulary of each categorical column)
        # -----------------------
        self.label_encoders = {}
        for col in self.categorical_columns:
            le = LabelEncoder()
            
            # Collect original categories + special tokens
            orig_vals = df[col].astype(str).unique()
            special_tokens = ["[MASK]", "[NULL]"]
            categories = np.unique(np.concatenate([orig_vals, special_tokens]))
            
            # Fit with everything (original values + [MASK], [NULL])
            le.fit(categories)
            self.label_encoders[col] = le
        
        # -----------------------
        # 2) Categorical Embeddings
        # -----------------------
        self.num_categories = {
            col: len(self.label_encoders[col].classes_)
            for col in self.categorical_columns
        }
        
        self.embedding_layers = nn.ModuleDict({
            re.sub(r'[^a-zA-Z0-9_]', '_', col): nn.Embedding(self.num_categories[col], self.dimensao)
            for col in self.categorical_columns
        })
        
        # -----------------------
        # 3) MLP for Numerical columns
        # -----------------------
        self.mlp_layers = nn.ModuleDict({
            self.column_to_key[col]: nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dimensao)
            ) for col in self.numerical_columns
        })
        
        # -----------------------
        # 4) Special Embeddings for Numerical columns
        # -----------------------
        self.special_embeddings = nn.ParameterDict()
        for col in self.numerical_columns:
            safe_key = self.column_to_key[col]
            self.special_embeddings[f"{safe_key}_mask"] = nn.Parameter(torch.randn(dimensao))
            self.special_embeddings[f"{safe_key}_null"] = nn.Parameter(torch.randn(dimensao))
        
        # -----------------------
        # 5) [CLS] Token
        # -----------------------
        self.cls_token = nn.Parameter(torch.randn(dimensao))
        
        # -----------------------
        # 6) Positional Embedding
        # -----------------------
        self.n_tokens = len(self.categorical_columns) + len(self.numerical_columns)
        self.pos_embedding_layer = nn.Embedding(self.n_tokens + 1, self.dimensao)

    def forward(self, df):
        """
        For each row in the DataFrame, generates the resulting embedding:
          1) Transforms each categorical column into indices and passes through nn.Embedding.
          2) For each numerical column:
               - If it's a real value, pass through MLP.
               - If it's [MASK], use the special "mask" embedding.
               - If it's [NULL], use the special "null" embedding.
          3) Concatenate embeddings of all columns (categorical+numerical).
          4) Reshape to (batch_size, n_tokens, dimension) and insert [CLS] at the top.
          5) Generate position indices and add position embedding, returning shape
             (batch_size, n_tokens+1, dimension).
        """
        dev = self.cls_token.device

        # =============== #
        # 1) EMBEDDINGS OF CATEGORICAL COLUMNS
        # =============== #
        cat_embeds_list = []
        for col in self.categorical_columns:
            col_values = df[col].astype(str).values
            indices = self.label_encoders[col].transform(col_values)
            indices_tensor = torch.tensor(indices, dtype=torch.long, device=dev)
            
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', col)
            col_emb = self.embedding_layers[safe_key](indices_tensor)
            cat_embeds_list.append(col_emb)
        
        if len(cat_embeds_list) > 0:
            cat_embeds = torch.cat(cat_embeds_list, dim=1)
        else:
            cat_embeds = None
        
        # =============== #
        # 2) EMBEDDINGS OF NUMERICAL COLUMNS
        # =============== #
        numeric_values, mask_flags, null_flags = split_numeric_and_special(df, self.numerical_columns, device=dev)
        
        num_embeds_list = []
        for j, col in enumerate(self.numerical_columns):
            column_values = numeric_values[:, j].unsqueeze(1)
            column_mask   = mask_flags[:, j]
            column_null   = null_flags[:, j]
            
            safe_key = self.column_to_key[col]
            
            # Pass through MLP
            mlp_output = self.mlp_layers[safe_key](column_values)
            col_final  = torch.zeros_like(mlp_output)
            
            # Identify masked / null / normal indices
            idx_mask   = (column_mask == True).nonzero(as_tuple=True)[0]
            idx_null   = (column_null == True).nonzero(as_tuple=True)[0]
            idx_normal = ((column_mask == False) & (column_null == False)).nonzero(as_tuple=True)[0]
            
            # Fill values
            col_final[idx_normal] = mlp_output[idx_normal]
            col_final[idx_mask]   = self.special_embeddings[f"{safe_key}_mask"]
            col_final[idx_null]   = self.special_embeddings[f"{safe_key}_null"]
            
            num_embeds_list.append(col_final)
        
        if len(num_embeds_list) > 0:
            num_embeds = torch.cat(num_embeds_list, dim=1)
        else:
            num_embeds = None
        
        # =============== #
        # 3) CONCATENATE CATEGORICAL + NUMERICAL
        # =============== #
        if cat_embeds is not None and num_embeds is not None:
            final_embeddings = torch.cat([cat_embeds, num_embeds], dim=1)
        elif cat_embeds is not None:
            final_embeddings = cat_embeds
        else:
            final_embeddings = num_embeds
        
        # =============== #
        # 4) INSERT [CLS] TOKEN AND RESHAPE
        # =============== #
        batch_size = final_embeddings.shape[0]
        
        n_tokens = 0
        if cat_embeds is not None:
            n_tokens += len(self.categorical_columns)
        if num_embeds is not None:
            n_tokens += len(self.numerical_columns)
        
        final_embeddings = final_embeddings.view(batch_size, n_tokens, self.dimensao)
        
        cls_token_expanded = self.cls_token.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, self.dimensao)
        final_embeddings = torch.cat([cls_token_expanded, final_embeddings], dim=1)
        
        # =============== #
        # 5) ADD POSITIONAL EMBEDDING
        # =============== #
        seq_len = final_embeddings.shape[1]
        pos_indices = torch.arange(seq_len, device=dev).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.pos_embedding_layer(pos_indices)
        
        final_embeddings = final_embeddings + pos_embeds
        
        return final_embeddings 