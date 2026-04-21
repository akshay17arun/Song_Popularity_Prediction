"""
Run this once from the project root before launching the demo:

    python demo/save_scaler.py

Reproduces the exact StandardScaler fit from the training notebook and saves it
to demo/scaler.pkl so app.py can apply identical normalisation at inference time.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

df = pd.read_csv(DATA / "all_tracks_clean.csv")
image_data = np.load(DATA / "spectrogram_tensors.npz")
indices = image_data["indices"]

df_aligned = df.iloc[indices].reset_index(drop=True)

genre_dummies = pd.get_dummies(df_aligned["genre"], prefix="genre").astype(np.float32)
n_genre_cols = genre_dummies.shape[1]  # 27

num_feats = df_aligned[["gain", "duration_sec", "num_contributors", "track_position"]].copy().astype(np.float32)
num_feats["explicit"] = df_aligned["explicit"].astype(np.float32)

meta_raw = pd.concat([genre_dummies, num_feats], axis=1).values.astype(np.float32)

y_raw = df_aligned["tier"].values.astype(int)
y = (y_raw - 1) // 2

idx_all = np.arange(len(y))
idx_train, _ = train_test_split(idx_all, test_size=0.3, random_state=6500)

meta_train = meta_raw[idx_train].copy()
scaler = StandardScaler()
scaler.fit(meta_train[:, n_genre_cols:])

out_path = Path(__file__).parent / "scaler.pkl"
with open(out_path, "wb") as f:
    pickle.dump(scaler, f)

print(f"Saved scaler to {out_path}")
print(f"  Feature order after genre one-hots: gain, duration_sec, num_contributors, track_position, explicit")
print(f"  mean_  = {scaler.mean_}")
print(f"  scale_ = {scaler.scale_}")
