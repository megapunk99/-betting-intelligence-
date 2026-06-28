"""
Fix the broken calibration block in robust_ensemble.py.
The old Step 3/4 code was partially removed, leaving cal_probs_dict empty.
"""
import sys

path = 'src/betting_intel/models/robust_ensemble.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the broken calibration block
old_block = '''        # \u2500\u2500 Step 4: Calibrated probabilities and Brier scores \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        cal_probs_dict: dict[str, np.ndarray] = {}
        self._raw_probs = np.column_stack([oos_dict[name] for name in oos_dict]) if len(oos_dict) > 1 \\
            else oos_dict[list(oos_dict.keys())[0]].reshape(-1, 1)

        for model_name, oos_probs in oos_dict.items():
            if model_name in s'''

new_block = '''        # \u2500\u2500 Step 3: Calibrate each model's OOS predictions (v6.6) \u2500\u2500\u2500\u2500
        self._calibrators = {}
        self._calibration_models = {}
        cal_probs_dict: dict[str, np.ndarray] = {}

        if self.calibrate:
            cal_probs_dict = self._calibrate_with_isotonic(oos_dict, oos_targets)
        else:
            cal_probs_dict = dict(oos_dict)

        # \u2500\u2500 Step 4: Build calibrated/raw prob arrays \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        self._raw_probs = np.column_stack([oos_dict[name] for name in oos_dict]) if len(oos_dict) > 1 \\
            else oos_dict[list(oos_dict.keys())[0]].reshape(-1, 1)'''

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    print("OK: Calibration block replaced successfully")
else:
    # Try without unicode box chars
    old_block_v2 = '''        # Step 4: Calibrated probabilities and Brier scores
        cal_probs_dict: dict[str, np.ndarray] = {}
        self._raw_probs = np.column_stack([oos_dict[name] for name in oos_dict]) if len(oos_dict) > 1 \\
            else oos_dict[list(oos_dict.keys())[0]].reshape(-1, 1)

        for model_name, oos_probs in oos_dict.items():
            if model_name in s'''
    if old_block_v2 in content:
        content = content.replace(old_block_v2, new_block, 1)
        print("OK: Calibration block replaced (v2)")
    else:
        print(f"FAILED: Could not find calibration block")
        # Debug: find what's near "Step 4"
        idx = content.find("Step 4: Calibrated")
        if idx >= 0:
            print(f"Found 'Step 4: Calibrated' at {idx}")
            print(f"Context: {content[idx:idx+200]}")
        idx = content.find("cal_probs_dict:")
        if idx >= 0:
            print(f"Found 'cal_probs_dict:' at {idx}")
            print(f"Context: {content[idx-50:idx+200]}")

# Also remove any remaining broken loop code from the old calibration
# Look for the broken for loop 
old_loop = '''        for model_name, oos_probs in oos_dict.items():
            if model_name in self._calibrators:
                X_cal = np.clip(oos_probs, 0.001, 0.999).reshape(-1, 1)
                cal_probs = self._calibrators[model_name].predict_proba(X_cal)[:, 1]


        self._calibrated_probs'''

new_after = '''
        self._calibrated_probs'''

if old_loop in content:
    content = content.replace(old_loop, new_after, 1)
    print("OK: Removed broken old calibration loop")
else:
    # Try shorter match
    old_loop_v2 = 'for model_name, oos_probs in oos_dict.items():\n            if model_name in s'
    if old_loop_v2 in content:
        print("WARNING: Old calibration loop still present but couldn't match full block")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
