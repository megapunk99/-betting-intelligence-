"""
Fix script: Add SVM auto-downsampling to robust_ensemble.py.

Changes:
1. Insert _DownsampledSVC wrapper class before RobustPredictionSystem
2. Add svm_max_samples param to __init__
3. Replace raw SVC with _DownsampledSVC in _get_model_specs()
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'src/betting_intel/models/robust_ensemble.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# ── 1. Insert _DownsampledSVC class before RobustPredictionSystem ──
wrapper = (
    "\n"
    "# ── v6.6: SVM Auto-Downsampling Wrapper ────────────────────────────────────\n"
    "\n"
    "class _DownsampledSVC:\n"
    '    """\n'
    "    SVC wrapper that auto-downsamples training data when dataset is too large.\n"
    "\n"
    "    SVM scales O(n\\u00b2) in the number of samples, making it prohibitively slow\n"
    "    on datasets larger than ~5,000 rows. This wrapper transparently downsamples\n"
    "    to `max_samples` rows before fitting, logging a warning so the user knows.\n"
    "    Downsampling preserves class balance via stratified random sampling.\n"
    "\n"
    "    All SVC attributes (predict, predict_proba, support_vectors_, etc.) are\n"
    "    proxied through to the underlying fitted model.\n"
    '    """\n'
    "\n"
    "    def __init__(self, max_samples: int = 3000, **svc_kwargs):\n"
    "        self.max_samples = max_samples\n"
    "        self._svc_kwargs = svc_kwargs\n"
    "        self._model: Any = None\n"
    "\n"
    "    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs) -> _DownsampledSVC:\n"
    "        n = len(X)\n"
    "        if n > self.max_samples:\n"
    "            from sklearn.model_selection import StratifiedShuffleSplit\n"
    "            sss = StratifiedShuffleSplit(\n"
    "                n_splits=1,\n"
    "                train_size=self.max_samples,\n"
    "                random_state=self._svc_kwargs.get('random_state', 42),\n"
    "            )\n"
    "            train_idx, _ = next(sss.split(X, y))\n"
    "            X_sample = X[train_idx]\n"
    "            y_sample = y[train_idx]\n"
    "            logger.info(\n"
    "                f'SVM auto-downsampled: {n} \u2192 {len(X_sample)} samples '\n"
    "                f'(O(n\\u00b2) scaling avoided)'\n"
    "            )\n"
    "        else:\n"
    "            X_sample, y_sample = X, y\n"
    "\n"
    "        from sklearn.svm import SVC\n"
    "        self._model = SVC(**self._svc_kwargs)\n"
    "        self._model.fit(X_sample, y_sample, **fit_kwargs)\n"
    "        return self\n"
    "\n"
    "    def predict(self, X: np.ndarray) -> np.ndarray:\n"
    "        if self._model is None:\n"
    "            raise ValueError('Model not fitted yet.')\n"
    "        return self._model.predict(X)\n"
    "\n"
    "    def predict_proba(self, X: np.ndarray) -> np.ndarray:\n"
    "        if self._model is None:\n"
    "            raise ValueError('Model not fitted yet.')\n"
    "        return self._model.predict_proba(X)\n"
    "\n"
    "    def __getattr__(self, name: str) -> Any:\n"
    "        \"\"\"Proxy any other attributes (support_vectors_, coef_, etc.) to the fitted model.\"\"\"\n"
    "        if name in ('_model', '_svc_kwargs', 'max_samples'):\n"
    "            raise AttributeError(name)\n"
    "        if self._model is not None and hasattr(self._model, name):\n"
    "            return getattr(self._model, name)\n"
    "        raise AttributeError(f\"'{type(self).__name__}' has no attribute '{name}'\")\n"
    "\n"
    "\n"
)

# Insert before the RobustPredictionSystem docstring
marker = 'class RobustPredictionSystem:'
idx = content.find(marker)
if idx == -1:
    print('ERROR: Could not find class RobustPredictionSystem')
    sys.exit(1)

# Find the blank line just before the class definition
insert_point = content.rfind('\n\n', 0, idx)
content = content[:insert_point + 1] + wrapper + content[insert_point + 1:]
print('1. _DownsampledSVC class inserted OK')

# ── 2. Add svm_max_samples to __init__ signature ──
old_init = '        use_svm: bool = True,               # v6.6 NEW'
new_init = '        use_svm: bool = True,               # v6.6 NEW\n        svm_max_samples: int = 3000,        # v6.6 NEW \u2014 auto-downsample for O(n\u00b2) scaling'
if old_init in content:
    content = content.replace(old_init, new_init, 1)
    print('2a. svm_max_samples param added to __init__ signature OK')
else:
    print('ERROR: Could not find use_svm line in __init__')
    sys.exit(1)

# Add self._svm_max_samples assignment
old_assign = '        self.use_svm = use_svm'
new_assign = '        self.use_svm = use_svm\n        self._svm_max_samples = svm_max_samples'
if old_assign in content:
    content = content.replace(old_assign, new_assign, 1)
    print('2b. self._svm_max_samples assignment added OK')
else:
    print('ERROR: Could not find self.use_svm = use_svm')
    sys.exit(1)

# ── 3. Replace SVC with _DownsampledSVC in _get_model_specs ──
# First try the full match with the comment
old_svm = (
    '            # Train on a sample if too large (SVM scales O(n\u00b2))\n'
    '                specs.append((\"SVM\", SVC, svm_params))'
)
new_svm = (
    '            # Auto-downsample to avoid O(n\u00b2) timeout on large datasets\n'
    '                svm_params["max_samples"] = self._svm_max_samples\n'
    '                specs.append((\"SVM\", _DownsampledSVC, svm_params))'
)

if old_svm in content:
    content = content.replace(old_svm, new_svm, 1)
    print('3. SVM spec updated to use _DownsampledSVC OK')
else:
    # Try without the comment
    alt_old = 'specs.append(("SVM", SVC, svm_params))'
    alt_new = 'svm_params["max_samples"] = self._svm_max_samples\n                specs.append(("SVM", _DownsampledSVC, svm_params))'
    if alt_old in content:
        content = content.replace(alt_old, alt_new, 1)
        print('3. SVM spec updated (alt match) OK')
    else:
        print('ERROR: Could not find SVM spec')
        sys.exit(1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('\nAll changes applied successfully!')
print(f'Final file size: {len(content)} chars')
