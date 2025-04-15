import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Load original CSV
df = pd.read_csv("songs_with_clusters_lyrics_features_selected_embeddings.csv")

# Load model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
if torch.cuda.is_available():
    model = model.to("cuda")

# Fill NaNs
df["summary"] = df["summary"].fillna("")
df["lyrics_audio_features_summary"] = df["lyrics_audio_features_summary"].fillna("")

# Compute and store embeddings as lists
df["summary_embedding"] = model.encode(df["summary"].tolist(), show_progress_bar=True).tolist()
df["audio_lyrics_embedding"] = model.encode(df["lyrics_audio_features_summary"].tolist(), show_progress_bar=True).tolist()

# Save with embeddings
df.to_parquet("songs_with_precomputed_embeddings.parquet", index=False)
