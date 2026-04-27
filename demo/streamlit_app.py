"""
Streamlit demo for the AST + Metadata Fusion song popularity predictor.

Run with:
    cd demo && streamlit run streamlit_app.py
"""

import os
import pickle
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Song Popularity Predictor",
    page_icon="🎵",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS — pink / green theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background-color: #0f0f14;
        color: #f0f0f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a24;
    }

    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, #ff6a92, #ff3d6e);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        width: 100%;
        transition: filter 0.2s;
    }
    .stButton > button:hover {
        filter: brightness(1.15);
        color: white;
    }

    /* Section cards */
    .card {
        background: #1a1a24;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        border: 1px solid #2a2a38;
    }

    /* Section headings */
    .section-heading {
        color: #ff6a92;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    /* Rating stars */
    .stars-row {
        display: flex;
        gap: 6px;
        justify-content: center;
        margin: 0.5rem 0 0.25rem;
    }
    .star {
        font-size: 2.6rem;
        line-height: 1;
    }
    .star-filled { color: #ff6a92; }
    .star-empty  { color: #3a3a4e; }

    /* Rating label */
    .rating-label {
        text-align: center;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 0.4rem;
    }
    .rating-sub {
        text-align: center;
        font-size: 0.88rem;
        color: #a0a0b8;
        margin-top: 0.1rem;
    }

    /* Divider */
    hr { border-color: #2a2a38; }

    /* Streamlit form labels */
    label { color: #d0d0e0 !important; font-size: 0.9rem !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #ff6a92 !important;
        border-radius: 12px !important;
        background: #13131c !important;
    }

    /* Number / select inputs */
    input[type="number"], select, .stSelectbox select {
        background-color: #13131c !important;
        border-color: #2a2a38 !important;
        color: #f0f0f0 !important;
        border-radius: 8px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #a0a0b8 !important;
        font-size: 0.85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEMO_DIR = Path(__file__).parent
MODEL_PATH = DEMO_DIR / "best_ast_model.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = DEMO_DIR.parent / "transformer_model" / "best_ast_model.pt"
SCALER_PATH = DEMO_DIR / "scaler.pkl"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENRES = [
    "African Music", "Alternative", "Asian Music", "Blues", "Brazilian Music",
    "Christian", "Classical", "Country", "Cumbia", "Dance", "Electro",
    "Films Games", "Folk", "Indian Music", "Jazz", "Kids", "Latin Music",
    "Metal", "Pop", "Rap Hip Hop", "Reggae", "Reggaeton", "RnB", "Rock",
    "Salsa", "Soul Funk", "Traditional Mexicano",
]

# Bucket 0 = most popular → rating 5; bucket 4 = least popular → rating 1
BUCKET_TO_RATING = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}

BUCKET_LABELS = {
    0: ("Tier 1–2", "Very High Popularity"),
    1: ("Tier 3–4", "High Popularity"),
    2: ("Tier 5–6", "Moderate Popularity"),
    3: ("Tier 7–8", "Low Chart Presence"),
    4: ("Tier 9–10", "Very Low Chart Presence"),
}

NUM_CLASSES = 5
META_DIM = 32

# ---------------------------------------------------------------------------
# Model architecture (verbatim from training notebook)
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


# ---------------------------------------------------------------------------
# CORN helpers
# ---------------------------------------------------------------------------

def corn_label_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    return (probas > 0.5).sum(dim=1)


def corn_proba(logits: torch.Tensor) -> np.ndarray:
    probas = torch.sigmoid(logits)
    cum = torch.cumprod(probas, dim=1)
    p = torch.zeros(logits.size(0), NUM_CLASSES)
    p[:, 0] = 1.0 - cum[:, 0]
    for k in range(1, NUM_CLASSES - 1):
        p[:, k] = cum[:, k - 1] - cum[:, k]
    p[:, NUM_CLASSES - 1] = cum[:, NUM_CLASSES - 2]
    p = p.clamp(min=0)
    p = p / p.sum(dim=1, keepdim=True)
    return p.detach().numpy()


# ---------------------------------------------------------------------------
# Load model & scaler (cached so they load once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    m = ASTFusionModel(meta_input_dim=META_DIM, num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    return m


@st.cache_resource(show_spinner=False)
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def audio_to_spectrogram(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=22050, duration=30, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_uint8 = cv2.normalize(mel_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mel_resized = cv2.resize(mel_uint8, (128, 128), interpolation=cv2.INTER_LINEAR)
    mel_final = mel_resized.astype(np.float32) / 255.0
    return mel_final[np.newaxis, np.newaxis, ...]  # (1, 1, 128, 128)


def build_metadata(genre, explicit, duration_sec, gain, num_contributors, track_position, scaler) -> np.ndarray:
    genre_vec = np.zeros(len(GENRES), dtype=np.float32)
    if genre in GENRES:
        genre_vec[GENRES.index(genre)] = 1.0
    numeric = np.array(
        [[gain, duration_sec, float(num_contributors), float(track_position), float(explicit)]],
        dtype=np.float32,
    )
    numeric_scaled = scaler.transform(numeric)
    return np.concatenate([genre_vec, numeric_scaled[0]])[np.newaxis, :]  # (1, 32)


def run_prediction(audio_path, genre, explicit, gain, num_contributors, track_position):
    model = load_model()
    scaler = load_scaler()

    y_raw, sr_raw = librosa.load(audio_path, sr=None, mono=True)
    duration_sec = float(len(y_raw) / sr_raw)

    spec = audio_to_spectrogram(audio_path)
    meta = build_metadata(genre, explicit, duration_sec, gain, num_contributors, track_position, scaler)

    spec_t = torch.tensor(spec, dtype=torch.float32)
    meta_t = torch.tensor(meta, dtype=torch.float32)

    with torch.no_grad():
        logits = model(spec_t, meta_t)

    probs = corn_proba(logits)[0]
    pred_bucket = int(np.argmax(probs))
    return pred_bucket, probs


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def stars_html(rating: int) -> str:
    filled = "★" * rating
    empty = "☆" * (5 - rating)
    stars = "".join(
        f'<span class="star star-filled">{s}</span>' for s in filled
    ) + "".join(
        f'<span class="star star-empty">{s}</span>' for s in empty
    )
    return f'<div class="stars-row">{stars}</div>'


def probability_chart(probs: np.ndarray) -> go.Figure:
    # Display from best (bucket 0) to worst (bucket 4), shown as rating 5→1
    labels = [f"{BUCKET_LABELS[i][1]}" for i in range(NUM_CLASSES)]
    ratings = [BUCKET_TO_RATING[i] for i in range(NUM_CLASSES)]
    top_bucket = int(np.argmax(probs))
    colors = ["#ff6a92" if i == top_bucket else "#00b76c" for i in range(NUM_CLASSES)]

    fig = go.Figure(
        go.Bar(
            x=[f"★{r}" for r in ratings],
            y=[float(p) for p in probs],
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
            hovertemplate="%{customdata}<br>Probability: %{y:.1%}<extra></extra>",
            customdata=labels,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d0d0e0", size=13),
        xaxis=dict(
            title="Predicted Rating",
            gridcolor="#2a2a38",
            tickfont=dict(size=14, color="#ff6a92"),
        ),
        yaxis=dict(
            title="Probability",
            gridcolor="#2a2a38",
            tickformat=".0%",
            range=[0, max(probs) * 1.25],
        ),
        margin=dict(t=20, b=10, l=10, r=10),
        height=300,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
        <span style="font-size:2.6rem;">🎵</span>
        <h1 style="color:#ff6a92; font-size:2rem; font-weight:900; margin:0.2rem 0 0;">
            Song Popularity Predictor
        </h1>
        <p style="color:#a0a0b8; font-size:0.95rem; margin-top:0.4rem;">
            Powered by Audio Spectrogram Transformer + Metadata Fusion
        </p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ── Audio upload ──────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-heading">🎧 Audio File</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload your song (MP3, WAV, FLAC, OGG)",
    type=["mp3", "wav", "flac", "ogg", "m4a"],
    label_visibility="collapsed",
)
if uploaded_file:
    st.audio(uploaded_file)
st.markdown("</div>", unsafe_allow_html=True)

# ── Song details ──────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-heading">🎼 Song Details</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    genre = st.selectbox("Genre", GENRES, index=GENRES.index("Pop"))
with col2:
    explicit = st.checkbox("Explicit Content", value=False)

st.markdown("</div>", unsafe_allow_html=True)

# ── Advanced options ──────────────────────────────────────────────────────────
with st.expander("⚙️  Advanced options (leave defaults if unknown)"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        gain = st.number_input("Gain (dB)", value=-8.5, step=0.5, format="%.1f")
    with col_b:
        num_contributors = st.number_input("Contributors", min_value=1, value=1, step=1)
    with col_c:
        track_position = st.number_input("Track Position", min_value=1, value=1, step=1)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("Predict Popularity ✨")

# ── Results ───────────────────────────────────────────────────────────────────
if predict_clicked:
    if uploaded_file is None:
        st.error("Please upload an audio file first.")
    else:
        with st.spinner("Analysing your track…"):
            with tempfile.NamedTemporaryFile(suffix=f".{uploaded_file.name.split('.')[-1]}", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                pred_bucket, probs = run_prediction(
                    tmp_path, genre, bool(explicit), float(gain),
                    int(num_contributors), int(track_position),
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()
            finally:
                os.unlink(tmp_path)

        rating = BUCKET_TO_RATING[pred_bucket]
        tier_label, description = BUCKET_LABELS[pred_bucket]

        st.markdown('<hr>', unsafe_allow_html=True)

        # Star display
        st.markdown(
            f"""
            <div class="card" style="text-align:center;">
                <p class="section-heading" style="text-align:center;">✨ Predicted Rating</p>
                {stars_html(rating)}
                <div class="rating-label" style="color:#ff6a92;">{rating} / 5</div>
                <div class="rating-sub">{description} &nbsp;·&nbsp; {tier_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-heading">📊 Probability Breakdown</p>', unsafe_allow_html=True)
        st.plotly_chart(probability_chart(probs), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:#4a4a60; font-size:0.78rem; margin-top:1rem;">
        AST Fusion · 4.9M params · trained on 7,549 Deezer chart tracks across 27 genres<br>
        Test accuracy: 43.9% exact &nbsp;|&nbsp; 83.9% within-1-bucket
    </p>
    """,
    unsafe_allow_html=True,
)
