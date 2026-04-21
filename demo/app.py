"""
Gradio demo for the AST + Metadata Fusion song popularity predictor.

Run locally:
    cd demo && python app.py

Deploy to HF Spaces:
    - Copy best_ast_model.pt and scaler.pkl into this directory
    - Push to a Hugging Face Space with sdk: gradio
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import librosa
import cv2
import gradio as gr
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — works both locally (model lives one level up) and on HF Spaces
# ---------------------------------------------------------------------------
DEMO_DIR = Path(__file__).parent
MODEL_PATH = DEMO_DIR / "best_ast_model.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = DEMO_DIR.parent / "transformer_model" / "best_ast_model.pt"

SCALER_PATH = DEMO_DIR / "scaler.pkl"

# ---------------------------------------------------------------------------
# Genre list — must match the alphabetically-sorted get_dummies output exactly
# ---------------------------------------------------------------------------
GENRES = [
    "African Music", "Alternative", "Asian Music", "Blues", "Brazilian Music",
    "Christian", "Classical", "Country", "Cumbia", "Dance", "Electro",
    "Films Games", "Folk", "Indian Music", "Jazz", "Kids", "Latin Music",
    "Metal", "Pop", "Rap Hip Hop", "Reggae", "Reggaeton", "RnB", "Rock",
    "Salsa", "Soul Funk", "Traditional Mexicano",
]

BUCKET_LABELS = {
    0: ("Tier 1-2", "Very High Popularity"),
    1: ("Tier 3-4", "High Popularity"),
    2: ("Tier 5-6", "Moderate Popularity"),
    3: ("Tier 7-8", "Low Chart Presence"),
    4: ("Tier 9-10", "Very Low Chart Presence"),
}

NUM_CLASSES = 5
META_DIM = 32  # 27 genre one-hots + gain, duration_sec, num_contributors, track_position, explicit

# ---------------------------------------------------------------------------
# Model architecture (copied verbatim from training notebook)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=1, d_model=256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, D, nh, nw = x.shape
        return x.flatten(2).transpose(1, 2)


class AudioSpectrogramTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=1, d_model=256,
                 num_heads=4, num_layers=6, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.norm(self.transformer(x))
        return x[:, 0]


class MetadataMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ASTFusionModel(nn.Module):
    def __init__(self, meta_input_dim, num_classes, ast_d_model=256, meta_output_dim=64):
        super().__init__()
        self.ast = AudioSpectrogramTransformer(d_model=ast_d_model)
        self.meta_mlp = MetadataMLP(meta_input_dim, output_dim=meta_output_dim)
        fusion_dim = ast_d_model + meta_output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes - 1),
        )

    def forward(self, spec, meta):
        audio_feat = self.ast(spec)
        meta_feat = self.meta_mlp(meta)
        fused = torch.cat([audio_feat, meta_feat], dim=1)
        return self.fusion_head(fused)


def corn_label_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    return (probas > 0.5).sum(dim=1)


def corn_proba(logits: torch.Tensor) -> np.ndarray:
    """Return probability for each of the 5 buckets from CORN logits."""
    probas = torch.sigmoid(logits)
    cum = torch.cumprod(probas, dim=1)  # P(Y >= k) for k=1..4
    p = torch.zeros(logits.size(0), NUM_CLASSES)
    p[:, 0] = 1.0 - cum[:, 0]
    for k in range(1, NUM_CLASSES - 1):
        p[:, k] = cum[:, k - 1] - cum[:, k]
    p[:, NUM_CLASSES - 1] = cum[:, NUM_CLASSES - 2]
    p = p.clamp(min=0)
    p = p / p.sum(dim=1, keepdim=True)  # renormalise to sum=1
    return p.detach().numpy()


# ---------------------------------------------------------------------------
# Load model and scaler once at startup
# ---------------------------------------------------------------------------

device = torch.device("cpu")

model = ASTFusionModel(meta_input_dim=META_DIM, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------------------------------------------
# Audio → spectrogram (mirrors load_spectrogram_data.py exactly)
# ---------------------------------------------------------------------------

def audio_to_spectrogram(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=22050, duration=30, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_uint8 = cv2.normalize(mel_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mel_resized = cv2.resize(mel_uint8, (128, 128), interpolation=cv2.INTER_LINEAR)
    mel_final = mel_resized.astype(np.float32) / 255.0
    return mel_final[np.newaxis, np.newaxis, ...]  # (1, 1, 128, 128)


# ---------------------------------------------------------------------------
# Build metadata vector
# ---------------------------------------------------------------------------

def build_metadata(genre: str, explicit: bool, duration_sec: float,
                   gain: float, num_contributors: int, track_position: int) -> np.ndarray:
    genre_vec = np.zeros(len(GENRES), dtype=np.float32)
    if genre in GENRES:
        genre_vec[GENRES.index(genre)] = 1.0

    # order must match training: gain, duration_sec, num_contributors, track_position, explicit
    numeric = np.array([[gain, duration_sec, float(num_contributors),
                         float(track_position), float(explicit)]], dtype=np.float32)
    numeric_scaled = scaler.transform(numeric)

    return np.concatenate([genre_vec, numeric_scaled[0]])[np.newaxis, :]  # (1, 32)


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def predict(audio_path, genre, explicit, gain, num_contributors, track_position):
    if audio_path is None:
        return "Please upload an audio file.", {}

    try:
        # Audio duration for metadata
        y_raw, sr_raw = librosa.load(audio_path, sr=None, mono=True)
        duration_sec = float(len(y_raw) / sr_raw)

        spec = audio_to_spectrogram(audio_path)
        meta = build_metadata(genre, explicit, duration_sec, gain, num_contributors, track_position)

        spec_t = torch.tensor(spec, dtype=torch.float32)
        meta_t = torch.tensor(meta, dtype=torch.float32)

        with torch.no_grad():
            logits = model(spec_t, meta_t)

        pred_bucket = int(corn_label_from_logits(logits).item())
        probs = corn_proba(logits)[0]

        tier_label, description = BUCKET_LABELS[pred_bucket]
        result = f"**{description}** ({tier_label})"

        label_probs = {
            f"{BUCKET_LABELS[i][1]} ({BUCKET_LABELS[i][0]})": float(probs[i])
            for i in range(NUM_CLASSES)
        }

        return result, label_probs

    except Exception as e:
        return f"Error: {e}", {}


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Song Popularity Predictor") as demo:
    gr.Markdown("# Song Popularity Predictor")
    gr.Markdown(
        "Upload an audio clip and fill in a few details. "
        "The model (Audio Spectrogram Transformer + Metadata Fusion) will predict "
        "which Deezer chart popularity tier your song falls into."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Audio File (MP3 or WAV)")
            genre_input = gr.Dropdown(choices=GENRES, value="Pop", label="Genre")
            explicit_input = gr.Checkbox(label="Explicit Content", value=False)

            with gr.Accordion("Advanced (optional)", open=False):
                gain_input = gr.Number(value=-8.5, label="Gain (dB)  — leave default if unknown")
                contributors_input = gr.Number(value=1, precision=0, label="Number of Contributors")
                position_input = gr.Number(value=1, precision=0, label="Track Position on Album")

            predict_btn = gr.Button("Predict Popularity", variant="primary")

        with gr.Column():
            result_text = gr.Markdown(label="Prediction")
            result_label = gr.Label(label="Bucket Probabilities", num_top_classes=5)

    predict_btn.click(
        fn=predict,
        inputs=[audio_input, genre_input, explicit_input, gain_input, contributors_input, position_input],
        outputs=[result_text, result_label],
    )

    gr.Markdown(
        "**Model:** AST Fusion (4.9M params) — Audio Spectrogram Transformer + Metadata MLP, "
        "trained on 7,549 Deezer chart tracks across 27 genres. "
        "Test accuracy: 43.9% exact, 83.9% within-1-bucket."
    )

if __name__ == "__main__":
    demo.launch()
