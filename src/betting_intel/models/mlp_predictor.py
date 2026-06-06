"""
MLP Neural Network Predictor — inspired by NBA_AI's architecture.

Architecture (matching NBA_AI):
  Linear(n_features, 256) → BatchNorm → GELU → Dropout
  → Linear(256, 128) → BatchNorm → GELU → Dropout
  → Linear(128, 64) → BatchNorm → GELU → Dropout
  → Linear(64, 1)

Loss: Huber (regression) or CrossEntropy (classification)
Has spread prediction with uncertainty (sigma output) like NBA_AI's L4 model.

Usage:
    predictor = MLPPredictor(input_dim=100, prediction_type="regression")
    predictor.fit(X_train, y_train)
    preds = predictor.predict(X_test)
    preds_with_uncertainty = predictor.predict_with_uncertainty(X_test)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


class MLPNetwork(nn.Module):
    """
    Multi-Layer Perceptron with batch normalization.

    Architecture (256 → 128 → 64):
      - Layer 1: Linear(n_features, 256) → BatchNorm → GELU → Dropout(0.2)
      - Layer 2: Linear(256, 128) → BatchNorm → GELU → Dropout(0.2)
      - Layer 3: Linear(128, 64) → BatchNorm → GELU → Dropout(0.1)
      - Output: Linear(64, n_outputs)

    For regression with uncertainty, outputs 2 values per sample:
      mu (predicted value) and log_sigma (log of uncertainty).
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 output_dim: int = 1, dropout: float = 0.2,
                 predict_uncertainty: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.predict_uncertainty = predict_uncertainty
        hidden_dims = hidden_dims or [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.features = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev_dim, output_dim)

        if predict_uncertainty:
            self.log_sigma_head = nn.Linear(prev_dim, output_dim)
            nn.init.constant_(self.log_sigma_head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) input features

        Returns:
            mu: (B, output_dim) predicted values
            If predict_uncertainty: also returns sigma: (B, output_dim)
        """
        h = self.features(x)
        mu = self.mu_head(h)

        if self.predict_uncertainty:
            log_sigma = self.log_sigma_head(h)
            sigma = F.softplus(log_sigma) + 1e-6
            return mu, sigma
        return mu


class MLPPredictor:
    """
    MLP Neural Network predictor with sklearn-compatible API.

    Features:
      - PyTorch backend with GPU support when available
      - 256→128→64 architecture with BatchNorm and GELU activations
      - Optional uncertainty prediction (sigma output)
      - Early stopping with patience
      - Learning rate scheduling
      - Automatic device selection (CUDA / CPU)

    Usage:
        predictor = MLPPredictor(input_dim=100, prediction_type="regression")
        predictor.fit(X_train, y_train)
        preds = predictor.predict(X_test)
    """

    def __init__(self, input_dim: int = 0, prediction_type: str = "regression",
                 hidden_dims: List[int] = None, dropout: float = 0.2,
                 learning_rate: float = 1e-3, batch_size: int = 64,
                 max_epochs: int = 200, patience: int = 15,
                 predict_uncertainty: bool = False):
        self.input_dim = input_dim
        self.prediction_type = prediction_type
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.predict_uncertainty = predict_uncertainty

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[MLPNetwork] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.is_fitted = False
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_epoch: int = 0
        self.is_classifier = prediction_type in ("classification", "binary")

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> "MLPPredictor":
        """
        Fit the MLP model to training data.

        Args:
            X: (n_samples, n_features) training features
            y: (n_samples,) training targets
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets for early stopping
            sample_weight: Optional sample weights

        Returns:
            self
        """
        if self.input_dim == 0:
            self.input_dim = X.shape[1]

        # Normalize features
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0).clip(min=1e-8)
        X_norm = (X - self.scaler_mean) / self.scaler_std

        # Convert to tensors
        X_t = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        # Handle classification targets (ensure 0/1)
        if self.is_classifier:
            y_t = y_t.clamp(0, 1).long()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            Xv_norm = (X_val - self.scaler_mean) / self.scaler_std
            Xv_t = torch.tensor(Xv_norm, dtype=torch.float32, device=self.device)
            yv_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
            if self.is_classifier:
                yv_t = yv_t.clamp(0, 1).long()
            val_dataset = TensorDataset(Xv_t, yv_t)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Build model
        output_dim = 2 if self.is_classifier else 1
        self.model = MLPNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim if not self.is_classifier else 2,
            dropout=self.dropout,
            predict_uncertainty=self.predict_uncertainty,
        ).to(self.device)

        # Loss
        if self.is_classifier:
            criterion = nn.CrossEntropyLoss()
        elif self.predict_uncertainty:
            criterion = self._gaussian_nll_loss
        else:
            criterion = nn.HuberLoss(delta=1.0)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        self.train_losses = []
        self.val_losses = []

        self.model.train()
        for epoch in range(self.max_epochs):
            # Training
            epoch_loss = 0.0
            n_batches = 0
            self.model.train()

            for batch_X, batch_y in loader:
                optimizer.zero_grad()

                if self.is_classifier:
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                elif self.predict_uncertainty:
                    mu, sigma = self.model(batch_X)
                    loss = criterion(mu, sigma, batch_y)
                else:
                    pred = self.model(batch_X)
                    loss = criterion(pred.squeeze(), batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)

            # Validation
            val_loss = float("inf")
            if val_loader is not None:
                self.model.eval()
                val_loss_total = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        if self.is_classifier:
                            logits = self.model(batch_X)
                            loss = criterion(logits, batch_y)
                        elif self.predict_uncertainty:
                            mu, sigma = self.model(batch_X)
                            loss = criterion(mu, sigma, batch_y)
                        else:
                            pred = self.model(batch_X)
                            loss = criterion(pred.squeeze(), batch_y)
                        val_loss_total += loss.item()
                        val_batches += 1
                val_loss = val_loss_total / max(val_batches, 1)
                self.val_losses.append(val_loss)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in
                                  self.model.state_dict().items()}
                    self.best_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
            else:
                scheduler.step(avg_loss)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Args:
            X: (n_samples, n_features) input features

        Returns:
            (n_samples,) 1-D array of predicted values
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        self.model.eval()
        X_norm = (X - self.scaler_mean) / self.scaler_std
        X_t = torch.tensor(X_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.is_classifier:
                logits = self.model(X_t)
                probs = F.softmax(logits, dim=1)
                return probs[:, 1].cpu().numpy()
            elif self.predict_uncertainty:
                mu, _ = self.model(X_t)
                return mu.cpu().numpy().ravel()
            else:
                pred = self.model(X_t)
                return pred.cpu().numpy().ravel()

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty (sigma) estimates.

        Args:
            X: (n_samples, n_features) input features

        Returns:
            mu: (n_samples,) predicted values
            sigma: (n_samples,) uncertainty estimates
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        if not self.predict_uncertainty:
            return self.predict(X), np.ones(len(X)) * np.nan

        self.model.eval()
        X_norm = (X - self.scaler_mean) / self.scaler_std
        X_t = torch.tensor(X_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            mu, sigma = self.model(X_t)
            return mu.cpu().numpy().ravel(), sigma.cpu().numpy().ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for binary classification).

        Args:
            X: (n_samples, n_features) input features

        Returns:
            (n_samples, 2) class probabilities
        """
        if not self.is_classifier:
            raise ValueError("predict_proba only available for classification")

        self.model.eval()
        X_norm = (X - self.scaler_mean) / self.scaler_std
        X_t = torch.tensor(X_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            return F.softmax(logits, dim=1).cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: (n_samples, n_features) test features
            y: (n_samples,) test targets

        Returns:
            Dict of metrics
        """
        preds = self.predict(X)
        metrics = {
            "mae": float(np.mean(np.abs(preds - y))),
            "rmse": float(np.sqrt(np.mean((preds - y) ** 2))),
            "n_samples": len(preds),
        }

        if not self.is_classifier:
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            metrics["r2"] = float(1 - ss_res / max(ss_tot, 1e-10))
            metrics["mean_pred"] = float(np.mean(preds))
            metrics["mean_actual"] = float(np.mean(y))
            metrics["bias"] = float(np.mean(preds - y))
        else:
            from sklearn.metrics import accuracy_score, roc_auc_score
            binary_preds = (preds > 0.5).astype(int)
            metrics["accuracy"] = float(accuracy_score(y, binary_preds))
            try:
                metrics["auc_roc"] = float(roc_auc_score(y, preds))
            except Exception:
                metrics["auc_roc"] = 0.5

        return metrics

    @staticmethod
    def _gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor,
                            y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss for Gaussian distribution.

        Loss = 0.5 * log(2*pi*sigma^2) + 0.5 * (y - mu)^2 / sigma^2
        """
        return (0.5 * torch.log(2 * torch.pi * sigma.pow(2))
                + 0.5 * (y - mu.squeeze()).pow(2) / sigma.pow(2)).mean()

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "name": "MLPPredictor",
            "prediction_type": self.prediction_type,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "device": str(self.device),
            "is_fitted": self.is_fitted,
            "best_epoch": self.best_epoch,
            "predict_uncertainty": self.predict_uncertainty,
        }


# ── Spread Predictor with Uncertainty (like NBA_AI L4) ──────────────────


class SpreadPredictorWithUncertainty(MLPPredictor):
    """
    Spread predictor that outputs both the predicted margin AND uncertainty.

    Matches NBA_AI's L4 approach where spread_sigma is conditioned on
    the predicted total (higher scoring games have more spread uncertainty).

    Architecture:
      Shared backbone (256→128→64) → 3 heads:
        - spread_mu: predicted margin (positive = home favored)
        - spread_sigma: uncertainty (higher = less confident)
        - win_prob: P(home wins)

    Usage:
        predictor = SpreadPredictorWithUncertainty(input_dim=100)
        predictor.fit(X_train, y_margin, y_winner)
        result = predictor.predict_with_probs(X_test)
        # result contains: spread_mu, spread_sigma, win_prob
    """

    def __init__(self, input_dim: int = 0, hidden_dims: List[int] = None,
                 dropout: float = 0.2, **kwargs):
        hidden_dims = hidden_dims or [256, 128, 64]
        super().__init__(
            input_dim=input_dim,
            prediction_type="regression",
            hidden_dims=hidden_dims,
            dropout=dropout,
            predict_uncertainty=True,
            **kwargs
        )
        self._margin_scaler_mean = None
        self._margin_scaler_std = None

    def fit(self, X: np.ndarray, y_margin: np.ndarray,
            y_winner: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None,
            y_val_margin: Optional[np.ndarray] = None) -> "SpreadPredictorWithUncertainty":
        """
        Fit the spread predictor.

        Args:
            X: (n_samples, n_features) training features
            y_margin: (n_samples,) actual point margins
            y_winner: (n_samples,) optional 0/1 winner labels
            X_val: optional validation features
            y_val_margin: optional validation margins
        """
        # Normalize margin target to stabilize training
        self._margin_scaler_mean = float(np.mean(y_margin))
        self._margin_scaler_std = float(np.std(y_margin).clip(min=1.0))
        y_norm = (y_margin - self._margin_scaler_mean) / self._margin_scaler_std

        # Normalize validation margin targets if provided
        if y_val_margin is not None:
            y_val_norm = (y_val_margin - self._margin_scaler_mean) / self._margin_scaler_std
        else:
            y_val_norm = None
        super().fit(X, y_norm, X_val=X_val, y_val=y_val_norm)
        return self

    def predict_spread(self, X: np.ndarray) -> np.ndarray:
        """Predict point spread (margin).

        Args:
            X: (n_samples, n_features)

        Returns:
            (n_samples,) predicted margin (positive = home favored)
        """
        raw_preds = self.predict(X)
        return raw_preds * self._margin_scaler_std + self._margin_scaler_mean

    def predict_with_probs(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict spread with uncertainty and win probability.

        Like NBA_AI's L4 PredictionHeads, returns:
          - spread_mu: predicted margin
          - spread_sigma: uncertainty estimate
          - win_prob: P(home wins)

        Args:
            X: (n_samples, n_features)

        Returns:
            Dict with spread_mu, spread_sigma, win_prob arrays
        """
        mu, sigma = self.predict_with_uncertainty(X)
        spread_mu = mu * self._margin_scaler_std + self._margin_scaler_mean
        spread_sigma = sigma * self._margin_scaler_std

        # Win probability from margin distribution
        # P(home wins) = P(margin > 0) = Phi(spread_mu / spread_sigma)
        from scipy import stats
        win_prob = stats.norm.cdf(spread_mu / np.clip(spread_sigma, 1e-6, None))

        return {
            "spread_mu": spread_mu,
            "spread_sigma": spread_sigma,
            "win_prob": win_prob,
        }


# ── Ensemble Predictor (Enhanced) ────────────────────────────────────────


class EnhancedEnsemble:
    """
    Enhanced ensemble combining multiple model types with adaptive weights.

    Like NBA_AI's ensemble but more advanced:
      - Each model type gets a weight based on recent performance
      - Weights are updated using exponential decay of Brier scores
      - Supports averaging spreads in arithmetic space and win probs in
        log-odds space (matching NBA_AI's approach)
      - Reports consensus level and model contributions

    Models can be:
      - sklearn-compatible objects with .predict() and optionally .predict_proba()
      - MLPPredictor instances
      - Any object with .predict() returning numpy arrays

    Usage:
        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", ridge_model, model_type="regression")
        ensemble.add_model("mlp", mlp_model, model_type="regression")
        ensemble.add_model("xgboost", xgb_model, model_type="regression")

        preds = ensemble.predict(X_test)
        breakdown = ensemble.get_prediction_breakdown(X_test)
    """

    def __init__(self, log_odds_averaging: bool = True,
                 weight_decay: float = 0.95):
        """
        Args:
            log_odds_averaging: If True, average win probabilities in
                log-odds space (NBA_AI's approach). If False, arithmetic mean.
            weight_decay: Exponential decay factor for performance tracking.
                0.95 = recent 20 predictions matter most.
        """
        self.models: Dict[str, Any] = {}
        self.model_types: Dict[str, str] = {}  # "regression" or "classification"
        self.weights: Dict[str, float] = {}
        self.log_odds_averaging = log_odds_averaging
        self.weight_decay = weight_decay
        self._performance_history: Dict[str, List[float]] = {}
        self._n_predictions = 0

    def add_model(self, name: str, model_obj: Any,
                  model_type: str = "regression",
                  weight: Optional[float] = None) -> "EnhancedEnsemble":
        """Add a model to the ensemble.

        Args:
            name: Unique model name
            model_obj: Model object with .predict() method
            model_type: "regression" or "classification"
            weight: Initial weight (default: 1.0 / n_models)
        """
        self.models[name] = model_obj
        self.model_types[name] = model_type
        self.weights[name] = weight if weight is not None else 1.0
        self._performance_history[name] = []
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction.

        For regression: weighted average of individual predictions.
        For classification: weighted average of probabilities in log-odds space.

        Args:
            X: (n_samples, n_features) input features

        Returns:
            (n_samples,) ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble. Add models first.")

        # Normalize weights
        total_weight = sum(max(w, 0.01) for w in self.weights.values())
        norm_weights = {k: max(v, 0.01) / total_weight
                        for k, v in self.weights.items()}

        # Collect predictions from all models
        all_preds = {}
        for name, model in self.models.items():
            try:
                preds = model.predict(X)
                all_preds[name] = np.asarray(preds).flatten()
            except Exception as e:
                print(f"  ⚠  Ensemble: {name} failed: {e}")
                all_preds[name] = np.zeros(X.shape[0]) if not self._n_predictions else None

        # Weighted combination
        n_samples = X.shape[0]
        ensemble_pred = np.zeros(n_samples, dtype=float)

        valid_models = [(n, p) for n, p in all_preds.items()
                        if p is not None and len(p) == n_samples]
        if not valid_models:
            return ensemble_pred

        for name, preds in valid_models:
            weight = norm_weights.get(name, 0.0)
            if self.model_types.get(name) == "classification" and self.log_odds_averaging:
                # Log-odds averaging (NBA_AI's approach)
                eps = 1e-8
                log_odds = np.log(np.clip(preds, eps, 1 - eps) / np.clip(1 - preds, eps, 1 - eps))
                ensemble_pred += weight * log_odds
            else:
                ensemble_pred += weight * preds

        # Convert back from log-odds if needed
        has_classification = any(
            self.model_types.get(name) == "classification"
            for name, _ in valid_models
        )
        if has_classification and self.log_odds_averaging:
            ensemble_pred = 1.0 / (1.0 + np.exp(-ensemble_pred))

        self._n_predictions += 1
        return ensemble_pred

    def predict_with_breakdown(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict with full model-by-model breakdown.

        Returns:
            Dict with:
              - ensemble_pred: (n_samples,) final predictions
              - model_preds: dict of {name: predictions}
              - weights: dict of {name: weight}
              - consensus: float (0-1, how much models agree)
              - n_models: int
        """
        ensemble_pred = self.predict(X)
        model_preds = {}
        for name, model in self.models.items():
            try:
                model_preds[name] = np.asarray(model.predict(X)).flatten()
            except Exception:
                model_preds[name] = np.full(X.shape[0], np.nan)

        # Consensus = 1 - normalized variance
        valid_preds = [p for p in model_preds.values()
                       if not np.all(np.isnan(p))]
        if len(valid_preds) >= 2:
            stacked = np.column_stack(valid_preds)
            variances = np.nanvar(stacked, axis=1)
            consensus = float(np.mean(1.0 - np.clip(variances * 20, 0, 1)))
        else:
            consensus = 1.0

        total_weight = sum(self.weights.values())
        normalized_weights = {k: v / total_weight for k, v in self.weights.items()}

        return {
            "ensemble_pred": ensemble_pred,
            "model_preds": model_preds,
            "weights": normalized_weights,
            "consensus": consensus,
            "n_models": len(self.models),
        }

    def update_weights(self, predictions: List[np.ndarray],
                       actuals: np.ndarray, window: int = 50):
        """Update ensemble weights based on recent performance.

        Uses exponential decay of errors (lower error = higher weight).
        Like NBA_AI: weights are adapted based on out-of-sample performance.

        Args:
            predictions: List of (n_samples,) prediction arrays, one per model
            actuals: (n_samples,) actual outcomes
            window: How many recent predictions to consider
        """
        model_names = list(self.models.keys())
        for i, name in enumerate(model_names):
            if i >= len(predictions):
                continue
            errors = (predictions[i] - actuals) ** 2
            self._performance_history[name].extend(errors.tolist())

            # Keep only recent history
            if len(self._performance_history[name]) > window:
                self._performance_history[name] = \
                    self._performance_history[name][-window:]

            # Weighted average error with exponential decay
            if self._performance_history[name]:
                n = len(self._performance_history[name])
                decays = np.array([self.weight_decay ** (n - 1 - j)
                                   for j in range(n)])
                decays = decays / decays.sum()
                avg_error = np.average(self._performance_history[name],
                                       weights=decays)

                # Weight = inverse of error (lower error = higher weight)
                new_weight = 1.0 / (avg_error + 1e-8)
                self.weights[name] = float(np.clip(new_weight, 0.1, 10.0))

    def get_params(self) -> Dict[str, Any]:
        """Get ensemble parameters."""
        return {
            "name": "EnhancedEnsemble",
            "n_models": len(self.models),
            "model_names": list(self.models.keys()),
            "weights": self.weights.copy(),
            "log_odds_averaging": self.log_odds_averaging,
            "weight_decay": self.weight_decay,
        }
