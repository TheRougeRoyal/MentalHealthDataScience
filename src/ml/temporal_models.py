"""
Temporal model training for mental health risk assessment.

This module provides training functions for RNN (LSTM/GRU) and Temporal Fusion Transformer
models with sequence preparation and padding utilities.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """PyTorch dataset for sequence data."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Initialize SequenceDataset.
        
        Args:
            sequences: Array of shape (n_samples, seq_length, n_features)
            labels: Array of shape (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class RNNModel(nn.Module):
    """RNN model with LSTM or GRU architecture."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = 'lstm',
        bidirectional: bool = False
    ):
        """
        Initialize RNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of RNN layers
            dropout: Dropout rate
            rnn_type: 'lstm' or 'gru'
            bidirectional: Whether to use bidirectional RNN
        """
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # RNN layer
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:  # gru
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        
        # Output layer
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * multiplier, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # RNN forward pass
        if self.rnn_type == 'lstm':
            out, (hidden, cell) = self.rnn(x)
        else:  # gru
            out, hidden = self.rnn(x)
        
        # Use last hidden state
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Apply dropout and output layer
        out = self.dropout(hidden)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out


class TemporalModelTrainer:
    """Trains temporal models (RNN and Temporal Fusion Transformer)."""
    
    def __init__(
        self,
        model_dir: str = "models",
        random_state: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize TemporalModelTrainer.
        
        Args:
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
            device: Device for training ('cuda' or 'cpu')
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        id_col: str,
        timestamp_col: str,
        target_col: str,
        seq_length: int = 30,
        padding_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare sequences from time-series data with padding.
        
        Args:
            df: Input dataframe with time-series data
            id_col: Column name for individual ID
            timestamp_col: Column name for timestamp
            target_col: Column name for target variable
            seq_length: Length of sequences
            padding_value: Value for padding shorter sequences
            
        Returns:
            Tuple of (sequences, labels, feature_names)
        """
        logger.info(f"Preparing sequences with length {seq_length}...")
        
        # Get feature columns
        feature_cols = [
            col for col in df.columns
            if col not in [id_col, timestamp_col, target_col]
        ]
        
        sequences = []
        labels = []
        
        # Group by individual
        for individual_id, group in df.groupby(id_col):
            # Sort by timestamp
            group = group.sort_values(timestamp_col)
            
            # Extract features and target
            features = group[feature_cols].values
            target = group[target_col].iloc[-1]  # Use last target value
            
            # Pad or truncate sequence
            if len(features) < seq_length:
                # Pad with padding_value
                padding = np.full(
                    (seq_length - len(features), len(feature_cols)),
                    padding_value
                )
                features = np.vstack([padding, features])
            elif len(features) > seq_length:
                # Take last seq_length timesteps
                features = features[-seq_length:]
            
            sequences.append(features)
            labels.append(target)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        logger.info(
            f"Prepared {len(sequences)} sequences with shape {sequences.shape}"
        )
        
        return sequences, labels, feature_cols
    
    def train_rnn(
        self,
        X_train_seq: np.ndarray,
        y_train: np.ndarray,
        X_val_seq: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = 'lstm',
        bidirectional: bool = False,
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ) -> Tuple[RNNModel, Dict[str, Any]]:
        """
        Train RNN model with LSTM or GRU architecture.
        
        Args:
            X_train_seq: Training sequences (n_samples, seq_length, n_features)
            y_train: Training labels
            X_val_seq: Validation sequences
            y_val: Validation labels
            feature_names: List of feature names
            hidden_size: Hidden layer size
            num_layers: Number of RNN layers
            dropout: Dropout rate
            rnn_type: 'lstm' or 'gru'
            bidirectional: Whether to use bidirectional RNN
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Tuple of (trained_model, metadata_dict)
        """
        logger.info(f"Training {rnn_type.upper()} model...")
        
        # Create datasets
        train_dataset = SequenceDataset(X_train_seq, y_train)
        val_dataset = SequenceDataset(X_val_seq, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Initialize model
        input_size = X_train_seq.shape[2]
        model = RNNModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1)
                    
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Calculate validation metrics
        from sklearn.metrics import roc_auc_score
        model.eval()
        val_preds = []
        
        with torch.no_grad():
            for sequences, _ in val_loader:
                sequences = sequences.to(self.device)
                outputs = model(sequences)
                val_preds.extend(outputs.cpu().numpy())
        
        val_preds = np.array(val_preds).flatten()
        val_score = roc_auc_score(y_val, val_preds)
        
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Validation AUROC: {val_score:.4f}")
        
        # Create metadata
        metadata = {
            'model_type': f'rnn_{rnn_type}',
            'architecture': {
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'bidirectional': bidirectional
            },
            'training_params': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs_trained': epoch + 1
            },
            'best_val_loss': float(best_val_loss),
            'val_auroc': float(val_score),
            'sequence_length': X_train_seq.shape[1],
            'n_features': input_size,
            'feature_names': feature_names,
            'training_samples': len(X_train_seq),
            'trained_at': datetime.now().isoformat(),
            'random_state': self.random_state
        }
        
        # Save model and metadata
        model_path = self.model_dir / f"rnn_{rnn_type}_model.pt"
        metadata_path = self.model_dir / f"rnn_{rnn_type}_metadata.json"
        
        torch.save(model.state_dict(), model_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return model, metadata
    
    def train_temporal_fusion(
        self,
        df: pd.DataFrame,
        time_idx: str,
        target: str,
        group_ids: List[str],
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_known_categoricals: List[str],
        time_varying_known_reals: List[str],
        time_varying_unknown_reals: List[str],
        max_encoder_length: int = 30,
        max_prediction_length: int = 1,
        batch_size: int = 64,
        max_epochs: int = 50
    ) -> Tuple[TemporalFusionTransformer, Dict[str, Any]]:
        """
        Train Temporal Fusion Transformer using PyTorch Forecasting.
        
        Args:
            df: Input dataframe with time-series data
            time_idx: Column name for time index
            target: Column name for target variable
            group_ids: List of columns identifying time series
            static_categoricals: Static categorical features
            static_reals: Static real-valued features
            time_varying_known_categoricals: Time-varying known categorical features
            time_varying_known_reals: Time-varying known real-valued features
            time_varying_unknown_reals: Time-varying unknown real-valued features
            max_encoder_length: Maximum length of encoder
            max_prediction_length: Maximum prediction length
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs
            
        Returns:
            Tuple of (trained_model, metadata_dict)
        """
        logger.info("Training Temporal Fusion Transformer...")
        
        # Create TimeSeriesDataSet
        training = TimeSeriesDataSet(
            df,
            time_idx=time_idx,
            target=target,
            group_ids=group_ids,
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        
        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df,
            predict=True,
            stop_randomization=True
        )
        
        # Create dataloaders
        train_dataloader = training.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=0
        )
        val_dataloader = validation.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=0
        )
        
        # Initialize model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # 7 quantiles
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        
        # Train model
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='auto',
            enable_model_summary=True,
            gradient_clip_val=0.1
        )
        
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        # Create metadata
        metadata = {
            'model_type': 'temporal_fusion_transformer',
            'architecture': {
                'max_encoder_length': max_encoder_length,
                'max_prediction_length': max_prediction_length,
                'hidden_size': 16,
                'attention_head_size': 1,
                'dropout': 0.1
            },
            'training_params': {
                'batch_size': batch_size,
                'max_epochs': max_epochs
            },
            'group_ids': group_ids,
            'static_categoricals': static_categoricals,
            'static_reals': static_reals,
            'time_varying_known_categoricals': time_varying_known_categoricals,
            'time_varying_known_reals': time_varying_known_reals,
            'time_varying_unknown_reals': time_varying_unknown_reals,
            'training_samples': len(df),
            'trained_at': datetime.now().isoformat(),
            'random_state': self.random_state
        }
        
        # Save model and metadata
        model_path = self.model_dir / "tft_model.ckpt"
        metadata_path = self.model_dir / "tft_metadata.json"
        
        trainer.save_checkpoint(model_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return tft, metadata
