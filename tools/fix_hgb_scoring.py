"""
Fix script: Change HGB scoring param from 'log_loss' to 'neg_log_loss'.
'log_loss' is not a valid sklearn scoring name; 'neg_log_loss' is.
"""
path = 'src/betting_intel/models/robust_ensemble.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old = '"scoring": "log_loss"'
new = '"scoring": "neg_log_loss"'
if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Fixed: {old} -> {new}")
else:
    print(f"WARNING: Could not find '{old}' in file")
    if new in content:
        print("(already fixed)")
