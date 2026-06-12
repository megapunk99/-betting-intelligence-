#!/usr/bin/env python3
"""Step 1: Load NBA data, build features, save to disk for fast model training."""
import sys, os, warnings, joblib
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer

print("Loading NBA data...")
loader = NBADataLoader()
fe = FeatureEngineer()
raw_df = loader.load_game_logs()
games_df = loader.build_game_dataset(raw_df)
raw_df = loader.compute_rest_days(raw_df)
print(f"Building features...")
feature_df = fe.build_all_features(games_df, raw_df)
feature_cols = fe.select_features(feature_df)
clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols)//2).copy()
clean_df = clean_df.reset_index(drop=True)
X = clean_df[feature_cols].fillna(0).values
y = (clean_df["point_diff"].values > 0).astype(int)

out = os.path.join(PROJECT_ROOT, "data", "ml_training_data.joblib")
joblib.dump({"X": X, "y": y, "feature_cols": feature_cols}, out)
print(f"Saved: {out}")
print(f"  X: {X.shape}, y: {y.sum()}/{len(y)} wins, features: {len(feature_cols)}")
