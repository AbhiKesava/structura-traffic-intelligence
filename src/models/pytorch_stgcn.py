"""
PyTorch Spatio-Temporal Graph Convolutional Network (ST-GCN) for Traffic Forecasting.
Combines Graph Convolutions (spatial road network) with Temporal GRU/Conv layers.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from src.utils.config import MODELS_DIR, PREDICTION_HORIZONS_MIN


def calculate_normalized_laplacian(adj_matrix: np.ndarray) -> torch.Tensor:
    """
    Computes symmetric normalized adjacency: D^{-1/2} (A + I) D^{-1/2}
    """
    adj = adj_matrix + np.eye(adj_matrix.shape[0])
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return torch.tensor(norm_adj, dtype=torch.float32)


class GraphConvLayer(nn.Module):
    """Spatial Graph Convolution: H = A_norm * X * W + b"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, N, in_features)
        norm_adj: shape (N, N)
        Returns: shape (B, N, out_features)
        """
        # (B, N, in) -> (B, N, out)
        support = torch.matmul(x, self.weight)
        # (N, N) x (B, N, out) -> (B, N, out)
        out = torch.matmul(norm_adj, support) + self.bias
        return F.relu(out)


class SpatioTemporalGCN(nn.Module):
    """
    Spatio-Temporal Traffic Speed Forecaster:
    - Input: (B, T_in, N)
    - Spatial feature extraction across graph sensors using GCN
    - Temporal sequence modeling across time steps using GRU
    - Multi-horizon prediction head -> (B, 3, N)
    """
    def __init__(
        self,
        num_sensors: int = 207,
        history_steps: int = 12,
        num_horizons: int = 3,
        spatial_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.15
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.history_steps = history_steps
        self.num_horizons = num_horizons

        # 1. Spatial embedding per time step
        self.gcn1 = GraphConvLayer(in_features=1, out_features=spatial_dim)
        
        # 2. Temporal modeling using GRU
        # Input to GRU: (B * N, T_in, spatial_dim)
        self.temporal_gru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        self.dropout = nn.Dropout(dropout)

        # 3. Multi-horizon decoder head
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_horizons)
        )

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T_in, N)
            norm_adj: Normalized adjacency matrix (N, N)
            
        Returns:
            out: Predicted speeds of shape (B, num_horizons, N)
        """
        B, T, N = x.shape
        # Add feature dimension: (B, T, N, 1)
        x = x.unsqueeze(-1)
        
        # Apply GCN across all timesteps
        # Reshape to (B * T, N, 1)
        x_reshaped = x.view(B * T, N, 1)
        gcn_out = self.gcn1(x_reshaped, norm_adj)  # (B * T, N, spatial_dim)
        
        # Reshape for GRU: each sensor independently across time -> (B, T, N, spatial_dim)
        gcn_out = gcn_out.view(B, T, N, -1)
        # Permute to (B, N, T, spatial_dim) -> (B * N, T, spatial_dim)
        gru_in = gcn_out.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)
        
        # Run GRU
        gru_out, h_n = self.temporal_gru(gru_in)  # h_n[-1]: (B * N, hidden_dim)
        last_hidden = self.dropout(h_n[-1])       # (B * N, hidden_dim)
        
        # Predict horizons
        preds = self.fc_out(last_hidden)          # (B * N, num_horizons)
        
        # Reshape back to (B, N, num_horizons) -> permute to (B, num_horizons, N)
        preds = preds.view(B, N, self.num_horizons).permute(0, 2, 1)
        return preds


class STGCNTrainer:
    """Wrapper for training, evaluation, and serialization of ST-GCN."""
    def __init__(
        self,
        model: SpatioTemporalGCN,
        norm_adj: torch.Tensor,
        scaler_params: Optional[Dict[str, float]] = None,
        lr: float = 0.003,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.norm_adj = norm_adj.to(self.device)
        self.scaler_mean = scaler_params.get("mean", 0.0) if scaler_params else 0.0
        self.scaler_std = scaler_params.get("std", 1.0) if scaler_params else 1.0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.HuberLoss(delta=1.0)

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Normalize inputs & targets
            x_norm = (batch_x - self.scaler_mean) / self.scaler_std
            y_norm = (batch_y - self.scaler_mean) / self.scaler_std
            
            self.optimizer.zero_grad()
            preds = self.model(x_norm, self.norm_adj)
            loss = self.criterion(preds, y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item() * len(batch_x)
            count += len(batch_x)

        return total_loss / max(1, count)

    @torch.no_grad()
    def evaluate(self, dataloader) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds = []
        all_trues = []
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            x_norm = (batch_x - self.scaler_mean) / self.scaler_std
            preds_norm = self.model(x_norm, self.norm_adj)
            
            # Inverse transform to original speed (mph)
            preds = (preds_norm * self.scaler_std) + self.scaler_mean
            all_preds.append(preds.cpu().numpy())
            all_trues.append(batch_y.numpy())

        preds_cat = np.concatenate(all_preds, axis=0)
        trues_cat = np.concatenate(all_trues, axis=0)
        mae = float(np.mean(np.abs(preds_cat - trues_cat)))
        return mae, preds_cat, trues_cat

    @torch.no_grad()
    def predict(self, x_numpy: np.ndarray) -> np.ndarray:
        """
        x_numpy: shape (B, T_in, N) or (T_in, N)
        Returns: unscaled speeds of shape (B, num_horizons, N)
        """
        self.model.eval()
        if x_numpy.ndim == 2:
            x_numpy = np.expand_dims(x_numpy, axis=0)
            
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=self.device)
        x_norm = (x_tensor - self.scaler_mean) / self.scaler_std
        preds_norm = self.model(x_norm, self.norm_adj)
        preds = (preds_norm * self.scaler_std) + self.scaler_mean
        return np.clip(preds.cpu().numpy(), 0.0, 90.0)

    def save(self, filepath: Optional[Path] = None):
        if filepath is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = MODELS_DIR / "pytorch_stgcn.pt"
            
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "num_sensors": self.model.num_sensors,
            "history_steps": self.model.history_steps,
            "num_horizons": self.model.num_horizons
        }, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Optional[Path], adj_matrix: np.ndarray, device: Optional[str] = None):
        if filepath is None:
            filepath = MODELS_DIR / "pytorch_stgcn.pt"
            
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        norm_adj = calculate_normalized_laplacian(adj_matrix)
        
        model = SpatioTemporalGCN(
            num_sensors=checkpoint["num_sensors"],
            history_steps=checkpoint["history_steps"],
            num_horizons=checkpoint["num_horizons"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        scaler_params = {
            "mean": checkpoint["scaler_mean"],
            "std": checkpoint["scaler_std"]
        }
        return cls(model=model, norm_adj=norm_adj, scaler_params=scaler_params, device=device)
