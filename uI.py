import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

# --- HARDCODED INPUT ---
HARDCODED_DESCRIPTION = (
    "Acoustic pop music with a mix of fast-paced celebratory moments and more reflective, acoustic elements would be an ideal fit for this image. "
    "The genre's ability to balance energy and intimacy aligns well with the image's mood, while its often optimistic tone complements the theme of "
    "educational achievement and personal growth."
)
PARQUET_PATH = "songs_with_precomputed_embeddings.parquet"

# --- CACHED LOADERS ---
@st.cache_resource
def load_model():
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return model.to("cuda") if torch.cuda.is_available() else model

@st.cache_data
def load_song_data():
    return pd.read_parquet(PARQUET_PATH)

# --- FAST FAISS RECOMMENDER ---
def recommend_with_faiss(user_description, df, model):
    user_emb = model.encode(user_description, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy().astype("float32").reshape(1, -1)

    summary_matrix = np.vstack(df["summary_embedding"].values).astype("float32")
    audio_matrix = np.vstack(df["audio_lyrics_embedding"].values).astype("float32")

    index_summary = faiss.IndexFlatIP(summary_matrix.shape[1])
    index_summary.add(summary_matrix)
    sim_summary, idx_summary = index_summary.search(user_emb, 5)

    index_audio = faiss.IndexFlatIP(audio_matrix.shape[1])
    index_audio.add(audio_matrix)
    sim_audio, idx_audio = index_audio.search(user_emb, 5)

    top5_summary = df.iloc[idx_summary[0]].copy()
    top5_audio = df.iloc[idx_audio[0]].copy()

    top5_summary["rank"] = range(1, 6)
    top5_summary["summary_similarity"] = sim_summary[0]
    top5_audio["rank"] = range(1, 6)
    top5_audio["audio_lyrics_similarity"] = sim_audio[0]

    return (
        top5_summary[["rank", "name", "artists", "cluster_selected", "summary_similarity"]],
        top5_audio[["rank", "name", "artists", "cluster_selected", "audio_lyrics_similarity"]],
    )

# --- STREAMLIT UI ---
st.set_page_config(page_title="⚡ Fast Music Recommender", layout="wide")
st.title("🎧 Fast Music Recommender using FAISS + Precomputed Embeddings")

st.markdown("### 🔍 Using hardcoded scene description:")
st.info(HARDCODED_DESCRIPTION)

# Load model + data
model = load_model()
df = load_song_data()

# Run recommendation
with st.spinner("Matching your vibe with the best music..."):
    top_summary, top_audio = recommend_with_faiss(HARDCODED_DESCRIPTION, df, model)

# Display results
st.subheader("📖 Top 5 Songs Based on Summary")
st.table(top_summary)

st.subheader("🎧 Top 5 Songs Based on Lyrics + Audio Summary")
st.table(top_audio)
