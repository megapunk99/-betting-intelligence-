"""
Fix remaining issues in robust_ensemble.py:
1. _prune_ensemble: change 0.48 threshold to also check Brier > 0.25
2. _prune_ensemble: add self._weights.pop(name, None) 
3. _compute_bootstrap_uncertainty: remove unused model_specs
4. _compute_ensemble_diversity: add warning when scipy missing
"""
import sys

path = 'src/betting_intel/models/robust_ensemble.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# Fix 1: Pruning threshold - change from hardcoded 0.48 to also check Brier
old_prune = "            acc = float(np.mean((preds > 0.5) == oos_targets))\n            if acc < 0.48:\n                pruned.append(name)\n                # Mark as degraded\n                if name in self._model_diagnostics:\n                    self._model_diagnostics[name].status = \"failed\""
new_prune = """            acc = float(np.mean((preds > 0.5) == oos_targets))
            diag = self._model_diagnostics.get(name)
            # Prune only if model is both inaccurate AND poorly calibrated
            if acc < 0.48 and diag and diag.oos_brier > 0.25:
                pruned.append(name)
                self._weights.pop(name, None)
                # Mark as degraded
                if name in self._model_diagnostics:
                    self._model_diagnostics[name].status = \"failed\""""

if old_prune in content:
    content = content.replace(old_prune, new_prune, 1)
    changes += 1
    print("OK: Fix 1 - Pruning threshold now checks Brier > 0.25, weights cleaned up")
else:
    print("FAIL: Fix 1 - Could not find pruning threshold code")

# Fix 2: Add weight cleanup in the second pruning loop (redundant pairs)
old_prune2 = "                if brier_i <= brier_j:\n                    pruned.append(name_j)\n                    remaining.remove(name_j)\n                else:\n                    pruned.append(name_i)\n                    remaining.remove(name_i)"
new_prune2 = """                if brier_i <= brier_j:
                    pruned.append(name_j)
                    self._weights.pop(name_j, None)
                    remaining.remove(name_j)
                else:
                    pruned.append(name_i)
                    self._weights.pop(name_i, None)
                    remaining.remove(name_i)"""

if old_prune2 in content:
    content = content.replace(old_prune2, new_prune2, 1)
    changes += 1
    print("OK: Fix 2 - Added weight cleanup in redundant pair pruning")
else:
    print("FAIL: Fix 2 - Could not find redundant pair pruning code")

# Fix 3: Fix bootstrap uncertainty - remove unused model_specs
old_bootstrap = """        try:\n            n_samples = self.n_bootstrap_samples\n            n = len(X)\n            all_bootstrap_preds = np.zeros((n_samples, n))\n\n            model_specs = self._get_model_specs()\n\n            for b in range(n_samples):"""
new_bootstrap = """        try:\n            n_samples = self.n_bootstrap_samples\n            n = len(X)\n            all_bootstrap_preds = np.zeros((n_samples, n))\n\n            for b in range(n_samples):"""

if old_bootstrap in content:
    content = content.replace(old_bootstrap, new_bootstrap, 1)
    changes += 1
    print("OK: Fix 3 - Removed unused model_specs in bootstrap uncertainty")
else:
    print("FAIL: Fix 3 - Could not find bootstrap model_specs line")

# Fix 4: Add warning when scipy is not available in diversity computation
old_scipy = '            return {"diversity_score": 0.5, "details": "scipy not available"}'
new_scipy = '            logger.warning("scipy required for ensemble diversity computation")\n            return {"diversity_score": 0.5, "details": "scipy not available"}'

if old_scipy in content:
    content = content.replace(old_scipy, new_scipy, 1)
    changes += 1
    print("OK: Fix 4 - Added warning when scipy missing for diversity")
else:
    print("FAIL: Fix 4 - Could not find scipy-not-available line")

# Write back
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nTotal fixes applied: {changes}")
