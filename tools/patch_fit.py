"""
Patch RobustPredictionSystem.fit() to wire in v6.6 features.
"""
import re, sys

path = 'src/betting_intel/models/robust_ensemble.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

changes = []

# ============================
# 1. Add hyperparameter tuning after min_samples check
# ============================
old_tune = '''        if n < self.min_train_samples + self.min_test_samples:
            raise ValueError(
                f"Need at least {self.min_train_samples + self.min_test_samples} samples, "
                f"got {n}. Consider reducing min_train_samples."
            )

        # Determine fold boundaries (chronological)'''

new_tune = '''        if n < self.min_train_samples + self.min_test_samples:
            raise ValueError(
                f"Need at least {self.min_train_samples + self.min_test_samples} samples, "
                f"got {n}. Consider reducing min_train_samples."
            )

        # Step 0: Hyperparameter Tuning (v6.6)
        if self.use_hyperparameter_tuning:
            self._tune_hyperparameters(X, y, verbose=verbose)

        # Determine fold boundaries (chronological)'''

if old_tune in content:
    content = content.replace(old_tune, new_tune, 1)
    changes.append("Added hyperparameter tuning call")

# ============================
# 2. Replace old calibration with _calibrate_with_isotonic()
# ============================
old_cal = '''        # Step 3: Calibrate each model's OOS predictions
        self._calibrators = {}
        for model_name in oos_dict:
            try:
                oos_probs = oos_dict[model_name]
                oos_probs_clipped = np.clip(oos_probs, 0.001, 0.999)

                cal = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
                calibrator = CalibratedClassifierCV(estimator=cal, method="sigmoid", cv=3)

                X_cal = oos_probs_clipped.reshape(-1, 1)
                y_cal = oos_targets

                if len(np.unique(y_cal)) >= 2:
                    calibrator.fit(X_cal, y_cal)
                    self._calibrators[model_name] = calibrator
                else:
                    logger.debug(f"Model {model_name}: only one class in OOS skipping calibration")
            except Exception as e:
                logger.debug(f"Calibration failed for {model_name}: {e}")

        # Step 4: Calibrated probabilities and Brier scores
        cal_probs_dict: dict[str, np.ndarray] = {}
        self._raw_probs = np.column_stack([oos_dict[name] for name in oos_dict]) if len(oos_dict) > 1 \\
            else oos_dict[list(oos_dict.keys())[0]].reshape(-1, 1)

        for model_name, oos_probs in oos_dict.items():
            if model_name in self._calibrators:
                X_cal = np.clip(oos_probs, 0.001, 0.999).reshape(-1, 1)
                try:
                    cal_probs = self._calibrators[model_name].predict_proba(X_cal)[:, 1]'''

new_cal = '''        # Step 3: Calibrate each model's OOS predictions (v6.6)
        # Uses auto-detection: isotonic > Platt > none
        self._calibrators = {}
        self._calibration_models = {}
        cal_probs_dict: dict[str, np.ndarray] = {}

        if self.calibrate:
            cal_probs_dict = self._calibrate_with_isotonic(oos_dict, oos_targets)
        else:
            cal_probs_dict = dict(oos_dict)

        # Step 4: Build calibrated/raw prob arrays
        self._raw_probs = np.column_stack([oos_dict[name] for name in oos_dict]) if len(oos_dict) > 1 \\
            else oos_dict[list(oos_dict.keys())[0]].reshape(-1, 1)'''

if old_cal in content:
    content = content.replace(old_cal, new_cal, 1)
    changes.append("Replaced old calibration with _calibrate_with_isotonic()")
else:
    # Try with the exact unicode chars from the file
    old_cal_v2 = '''        # \u2500\u2500 Step 3: Calibrate each model's OOS predictions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        self._calibrators = {}
        for model_name in oos_dict:
            try:
                oos_probs = oos_dict[model_name]
                oos_probs_clipped = np.clip(oos_probs, 0.001, 0.999)

                cal = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
                calibrator = CalibratedClassifierCV(estimator=cal, method="sigmoid", cv=3)

                X_cal = oos_probs_clipped.reshape(-1, 1)
                y_cal = oos_targets

                if len(np.unique(y_cal)) >= 2:
                    calibrator.fit(X_cal, y_cal)
                    self._calibrators[model_name] = calibrator
                else:
                    logger.debug(f"Model {model_name}: only one class in OOS \u2014 skipping calibration")
            except Exception as e:
                logger.debug(f"Calibration failed for {model_name}: {e}")

        # Step 4: Calibrated probabilities and Brier scores
        cal_probs_dict: dict[str, np.ndarray] = {}
        self._raw_probs = np.column_stack([oos_dict[name] for name in oos_dict]) if len(oos_dict) > 1 \\
            else oos_dict[list(oos_dict.keys())[0]].reshape(-1, 1)

        for model_name, oos_probs in oos_dict.items():
            if model_name in self._calibrators:
                X_cal = np.clip(oos_probs, 0.001, 0.999).reshape(-1, 1)
                try:
                    cal_probs = self._calibrators[model_name].predict_proba(X_cal)[:, 1]'''
    if old_cal_v2 in content:
        content = content.replace(old_cal_v2, new_cal, 1)
        changes.append("Replaced old calibration (v2) with _calibrate_with_isotonic()")
    else:
        changes.append("FAILED: Could not find calibration block")

# ============================
# 3. Remove remaining old calibration loop code
# ============================
# After the replacement above, the code may have leftover: cal_probs_dict[model_name] = cal_probs
old_remnant = '                    cal_probs_dict[model_name] = cal_probs\n                except Exception:\n                    cal_probs_dict[model_name] = oos_probs\n            else:\n                cal_probs_dict[model_name] = oos_probs\n'
if old_remnant in content:
    content = content.replace(old_remnant, '\n', 1)
    changes.append("Removed old calibration loop remnant")

# ============================
# 4. Add post-training steps after Step 9
# ============================
old_end = '''        self._fitted = True
        self._fit_timestamp = datetime.now().isoformat()
        return self'''

new_end = '''        # Step 10: Adversarial Validation (v6.6)
        self._run_adversarial_validation(X, y, verbose=verbose)

        # Step 11: Ensemble Diversity and Pruning (v6.6)
        if self.pruning_keep_top_n > 0 and len(self._model_diagnostics) >= 3:
            self._compute_ensemble_diversity(oos_dict)
            self._prune_ensemble(oos_dict, oos_targets)
            # Recompute weights after pruning
            if self._pruned_models:
                remaining_weights = {
                    n: w for n, w in self._weights.items()
                    if n not in self._pruned_models
                }
                total = sum(remaining_weights.values())
                if total > 0:
                    self._weights = {n: w / total for n, w in remaining_weights.items()}

        # Step 12: Permutation Importance (v6.6)
        self._compute_permutation_importance(X, y, n_repeats=3, n_features=20, verbose=verbose)

        # Step 13: Bootstrap Uncertainty (v6.6)
        self._compute_bootstrap_uncertainty(X, y, verbose=verbose)

        self._fitted = True
        self._fit_timestamp = datetime.now().isoformat()
        return self'''

# Count occurrences before replacement
count_old_end = content.count(old_end)
if count_old_end == 1:
    content = content.replace(old_end, new_end, 1)
    changes.append("Added post-training steps")
elif count_old_end > 1:
    # There are multiple matches (fit() and MarketInefficiencySystem.fit())
    # Find the first occurrence (RobustPredictionSystem)
    idx = content.find(old_end)
    if idx > 0:
        content = content[:idx] + new_end + content[idx + len(old_end):]
        changes.append("Added post-training steps (targeted replacement)")

# ============================
# Write back
# ============================
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

sys.stdout.reconfigure(encoding='utf-8')
for c in changes:
    print(f"OK: {c}")
print(f"\nTotal changes: {len(changes)}")
