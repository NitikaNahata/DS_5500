# main.py (LLaVA + Dropdown for Reasoning Models + Music Recommendation + Ranked Output)

import sys
if sys.version_info >= (3, 9):
    from collections.abc import Sequence
    from typing_extensions import Buffer
    import collections.abc
    collections.abc.Buffer = Buffer
else:
    from collections.abc import Buffer

import streamlit as st
import ollama
import base64
import asyncio
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from configurations import (
    IMAGE_CAPTIONING_LLVM_PROMPT_V3,
    USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT
)

REASONING_MODELS = {
    "DeepSeek (default)": "deepseek-coder:6.7b",
    "Minicpm": "minicpm-v:8b",
    "Llama Intuitive Thinker": "mychen76/llama3.1-intuitive-thinker:chain-of-thoughts.q5"
}

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def initialize_session_states():
    keys = ["user_music_description", "user_uploaded_image", "llava_response"]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None

def get_deepseek_response(model_name, user_description, vision_description, prompt_template):
    formatted_prompt = prompt_template.format(
        user_description=user_description,
        vision_description=vision_description
    )
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": formatted_prompt}]
    )
    return response["message"]["content"]
##from sentence_transformers import SentenceTransformer
##from sklearn.metrics.pairwise import cosine_similarity
def recommend_songs(final_reasoning_text, csv_path="songs_with_clusters_lyrics_features_selected.csv"):
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    if torch.cuda.is_available():
        model = model.to('cuda')

    user_embedding = model.encode(final_reasoning_text, convert_to_tensor=True)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['lyrics_summary', 'lyrics_audio_features_summary'], how='all')
    df['lyrics_summary'] = df['lyrics_summary'].fillna("")
    df['lyrics_audio_features_summary'] = df['lyrics_audio_features_summary'].fillna("")

    lyrics_embeddings = model.encode(df['lyrics_summary'].tolist(), convert_to_tensor=True)
    audio_lyrics_embeddings = model.encode(df['lyrics_audio_features_summary'].tolist(), convert_to_tensor=True)

    lyrics_sim = cosine_similarity(user_embedding.cpu().numpy().reshape(1, -1), lyrics_embeddings.cpu().numpy())[0]
    audio_lyrics_sim = cosine_similarity(user_embedding.cpu().numpy().reshape(1, -1), audio_lyrics_embeddings.cpu().numpy())[0]

    df['lyrics_similarity'] = lyrics_sim
    df['audio_lyrics_similarity'] = audio_lyrics_sim

    top5_lyrics = df.sort_values(by='lyrics_similarity', ascending=False).head(5).copy()
    top5_audio_lyrics = df.sort_values(by='audio_lyrics_similarity', ascending=False).head(5).copy()

    top5_lyrics['rank'] = range(1, 6)
    top5_audio_lyrics['rank'] = range(1, 6)

    return top5_lyrics[['rank', 'name', 'artists', 'lyrics_similarity']], top5_audio_lyrics[['rank', 'name', 'artists', 'audio_lyrics_similarity']]


async def main():
    initialize_session_states()
    st.title("Music Recommendation using AI")

    st.session_state.user_uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.session_state.user_music_description = st.text_input("Describe what kind of music would you like to add:")
    selected_reasoning_model = st.selectbox("Choose a reasoning model:", list(REASONING_MODELS.keys()))

    if st.session_state.user_uploaded_image is not None:
        st.image(st.session_state.user_uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        if st.session_state.user_uploaded_image and st.session_state.user_music_description:
            with st.spinner("Analyzing..."):
                base64_image = encode_image(st.session_state.user_uploaded_image)

                # Run LLaVA for vision description
                llava_resp = ollama.chat(
                    model='llava:7b',
                    messages=[{'role': 'user','content': IMAGE_CAPTIONING_LLVM_PROMPT_V3,'images': [base64_image]}]
                )
                st.session_state.llava_response = llava_resp['message']['content']
                st.expander("Analysis Result LLaVA 7B:").write(st.session_state.llava_response)

                # Use selected reasoning model for description
                reasoning_model_name = REASONING_MODELS[selected_reasoning_model]
                final_reasoning = get_deepseek_response(
                    reasoning_model_name,
                    st.session_state.user_music_description,
                    st.session_state.llava_response,
                    USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT
                )
                st.expander(f"Reasoning Output ({selected_reasoning_model})").write(final_reasoning)

                # Recommend songs
                top5_lyrics, top5_audio_lyrics = recommend_songs(final_reasoning)

                st.title("\U0001F3A7 Top Song Recommendations")
                st.subheader("\U0001F539 Based on Lyrics Only")
                for _, row in top5_lyrics.iterrows():
                    st.markdown(f"**#{row['rank']}** {row['name']} by {row['artists']} — Similarity: {row['lyrics_similarity']:.3f}")

                st.subheader("\U0001F538 Based on Lyrics + Audio Features")
                for _, row in top5_audio_lyrics.iterrows():
                    st.markdown(f"**#{row['rank']}** {row['name']} by {row['artists']} — Similarity: {row['audio_lyrics_similarity']:.3f}")
        else:
            st.warning("Please upload an image and enter a prompt.")

if __name__ == "__main__":
    asyncio.run(main())
