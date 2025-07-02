import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re

from .embedder import TabularEmbedder
from .transformer import TabularTransformerEncoder

class TridentPretrainer(nn.Module):
    """
    Given a masked DataFrame, asks the Transformer to reconstruct,
    only at [MASK] positions, the same embedding vector that would exist
    if the column was not masked.

    The target is the embedding (detached) produced by the TabularEmbedder itself
    when it receives the original DF (without [MASK]).
    """
    def __init__(self, embedder: TabularEmbedder, transformer: TabularTransformerEncoder):
        super().__init__()
        self.embedder = embedder
        self.transformer = transformer
        self.d_model = embedder.dimensao
        self.eps = 1e-8   # to avoid division by zero in optional normalization

    def forward(self, df_masked: pd.DataFrame, df_original: pd.DataFrame):
        device = next(self.parameters()).device

        # 1) "Corrupted" embeddings (Transformer input)
        emb_in = self.embedder(df_masked)              # (B, L, d)
        # 2) "Pure" embeddings (target) — detach to avoid backprop through them
        emb_target = self.embedder(df_original).detach()  # (B, L, d)

        # 3) Pass through Transformer
        encoded = self.transformer(emb_in)             # (B, L, d)

        # 4) Build boolean mask of where [MASK] existed
        #    → shape (B, L‑1)   (ignoring CLS at column 0)
        mask_matrix = []
        for col in self.embedder.categorical_columns + self.embedder.numerical_columns:
            mask_matrix.append((df_masked[col] == "[MASK]").values[:, None])  # shape (B,1)
        mask_matrix = np.concatenate(mask_matrix, axis=1)                    # (B, L‑1)
        mask_tensor = torch.tensor(mask_matrix, device=device, dtype=torch.bool)

        # 5) Select only masked positions (flatten)
        enc_sel  = encoded[:, 1:, :][mask_tensor]      # (N_mask, d)
        tgt_sel  = emb_target[:, 1:, :][mask_tensor]   # (N_mask, d)
        #enc_sel = nn.functional.normalize(enc_sel, dim=-1)
        #tgt_sel = nn.functional.normalize(tgt_sel, dim=-1)
        if enc_sel.numel() == 0:          # no [MASK] in the batch
            print("a")
            return torch.tensor(0., device=device, requires_grad=True), {}

        # 6) Loss = MSE between vectors
        loss = nn.functional.mse_loss(enc_sel, tgt_sel)

        return loss, {"mse_embedding": loss.item()}


class TridentModel(nn.Module):
    """
    Unified model for the classification task:
      1) Generates tabular embeddings (TabularEmbedder).
      2) Passes through Transformer encoders (TabularTransformerEncoder).
      3) Takes [CLS] and passes it through a linear layer for classification (nn.Linear).
    """
    def __init__(self, embedder, transformer, num_labels=2, class_weights=None):
        """
        Parameters
        ----------
        embedder : TabularEmbedder
            Responsible for generating embeddings (dimensao = d_model).
        transformer : TabularTransformerEncoder
            Bidirectional transformer encoder.
        num_labels : int
            Number of classes for classification. If 2 => binary.
        class_weights : torch.Tensor, optional
            Class weights for CrossEntropy, if you want to handle imbalance.
        """
        super().__init__()
        self.embedder = embedder
        self.transformer = transformer
        self.num_labels = num_labels
        
        self.classifier = nn.Sequential(
            nn.Linear(embedder.dimensao, embedder.dimensao // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedder.dimensao // 2, num_labels),
        )
        
        # Store class weights if provided
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, df, labels=None):
        """
        df : pd.DataFrame
            Input DataFrame (possibly masked/null, but in fine-tuning usually without mask).
        labels : Tensor (optional), shape (batch_size,) with true classes, 
                 if we want to calculate CrossEntropy loss.

        Returns
        -------
        logits : torch.Tensor
            shape (batch_size, num_labels)
        loss (optional) : torch.Tensor
            If 'labels' is provided, also returns the CrossEntropyLoss.
        """
        x = self.embedder(df)                # (batch_size, seq_len, d_model)
        encoded_output = self.transformer(x) # (batch_size, seq_len, d_model)
        
        # Extract [CLS], which is at encoded_output[:, 0, :]
        cls_representation = encoded_output[:, 0, :]  # (batch_size, d_model)
        
        # Final classification
        logits = self.classifier(cls_representation)  # (batch_size, num_labels)
        
        loss = None
        if labels is not None:
            # If we have class weights, apply them in CrossEntropy
            if self.class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            return logits, loss
        
        return logits 