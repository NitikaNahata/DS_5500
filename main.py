# Workaround for "Buffer" import error in Python 3.9+
import sys
if sys.version_info >= (3, 9):
    from collections.abc import Sequence
    from typing_extensions import Buffer
    import collections.abc
    collections.abc.Buffer = Buffer
else:
    from collections.abc import Buffer
import torch

import streamlit as st
import ollama
import base64
import asyncio
from configurations import USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT_V2, FINAL_VISION_MODEL, FINAL_REASONING_MODEL, IMAGE_CAPTIONING_LLVM_PROMPT_V3
import re
from sentence_transformers import SentenceTransformer
import faiss
import os
import pandas as pd
import numpy as np


CSV_PATH = "songs_with_clusters_lyrics_features_selected_embeddings.csv"
PARQUET_PATH = "songs_with_precomputed_embeddings.parquet"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"


def extract_tags_content(content, tags_list):
    """
    Extract content from specified tags and return it as a formatted string.
    
    Args:
        content (str): The input text containing tagged content
        tags_list (list): List of tag names to extract
    
    Returns:
        str: Formatted string with all extracted content
    """
    result = []
    for tag in tags_list:
        # Regex pattern that handles potential malformed XML and duplicate tags
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, content, re.DOTALL)
        
        if matches:
            for match in matches:
                # Clean up the extracted content (remove leading/trailing whitespace)
                cleaned_content = match.strip()
                # Add the tagged content with a header to the result
                result.append(f"{cleaned_content}\n")
    
    # Join all extracted content with double line breaks for UI display
    return "\n".join(result)

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def initialize_session_states():
    if "user_music_description" not in st.session_state:
        st.session_state.user_music_description = None
    if "user_uploaded_image" not in st.session_state:
        st.session_state.user_uploaded_image = None
    if "final_vision_model_response" not in st.session_state:
        st.session_state.final_vision_model_response = None
    if "final_reasoning_model_response" not in st.session_state:
        st.session_state.final_reasoning_model_response = None
    if "recommendations_list" not in st.session_state:
        st.session_state.recommendations_list = None

# Function to extract thinking and content from response
def parse_response(response):
    message_content = response["message"]["content"]
    thinking_content = message_content.split("<think>")[1].split("</think>")[0].strip()
    response_content = message_content.split("</think>")[1].strip()
    return thinking_content, response_content

def get_deepseek_response(model_name, user_music_description, image_description, prompt_template):
    # Format the prompt with user inputs
    # Format the prompt with user inputs
    formatted_prompt = prompt_template.format(
        user_music_description=user_music_description,
        image_description=image_description
    )
        
    # Call the DeepSeek model using Ollama
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": formatted_prompt}
        ]
    )
    
    # Return the model's response
    return response["message"]["content"]

def load_model():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    if torch.cuda.is_available():
        try:
            model = model.to("cuda")
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e):
                # Use to_empty() as suggested in the error message
                model = model.to_empty(device="cuda")
            else:
                raise
    
    return model  # Add this return statement

def load_song_data():
    if not os.path.exists(PARQUET_PATH):
        df = pd.read_csv(CSV_PATH)
        df = df.dropna(subset=["summary", "lyrics_audio_features_summary"])

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if torch.cuda.is_available():
            model = model.to("cuda")

        df["summary_embedding"] = model.encode(df["summary"].tolist(), show_progress_bar=True).tolist()
        df["audio_lyrics_embedding"] = model.encode(df["lyrics_audio_features_summary"].tolist(), show_progress_bar=True).tolist()
        df.to_parquet(PARQUET_PATH, index=False)
        return df
    else:
        return pd.read_parquet(PARQUET_PATH)

def get_recommendations_list(reasoning_model_response):
    return ['SONG 1','SONG 1','SONG 1','SONG 1','SONG 1','SONG 1','SONG 1','SONG 1','SONG 1']

def recommend_with_faiss(user_description, df, model):
    
    # Create a mapping dictionary for cluster names
    cluster_names = {
        0: "Romantic, polished pop full of love, nostalgia, and global feel-good vibes.",
        1: "Dark, emotional, and raw — a sonic dive into angst, identity, and reflection.",
        2: "Healing through heartbreak — emotional yet uplifting pop you can still dance to",
        3: "Empowering anthems with global rhythms and bold energy to rise and move.",
        4: "High-energy, trap-pop bangers made for flexing, partying, and owning the room",
        5: "Festive, cozy holiday tunes to light up memories, joy, and seasonal magic."
    }
    
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

    # Map cluster numbers to names
    top5_summary["cluster_name"] = top5_summary["cluster_selected"].map(cluster_names)
    top5_audio["cluster_name"] = top5_audio["cluster_selected"].map(cluster_names)

    top5_summary["rank"] = range(1, 6)
    top5_summary["summary_similarity"] = sim_summary[0]
    top5_audio["rank"] = range(1, 6)
    top5_audio["audio_lyrics_similarity"] = sim_audio[0]

    return (
        top5_summary[["rank", "name", "artists", "cluster_name", "summary_similarity"]],
        top5_audio[["rank", "name", "artists", "cluster_name", "audio_lyrics_similarity"]],
    )

async def main():
    # Initialize session state for resume and job link if they don't exist
    # Load model + song data
    rec_model = load_model()
    song_df = load_song_data()
    
    initialize_session_states()

    st.title("Music Recommendation using AI")

    st.session_state.user_uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.session_state.user_music_description = st.text_input("Describe what kind of music would you like to add:")

    if st.session_state.user_uploaded_image is not None:
        # Create a column layout with 3 columns
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # Display the image in the middle column with a specific width
        with col2:
            st.image(
                st.session_state.user_uploaded_image, 
                caption="Uploaded Image",
                width=300  # Set a specific width in pixels
            )

    #if st.session_state.user_uploaded_image is not None:
     #   st.image(st.session_state.user_uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        if st.session_state.user_uploaded_image is not None and st.session_state.user_music_description:
            with st.spinner("Recommending songs..."):
                base64_image = encode_image(st.session_state.user_uploaded_image)
                
                ## Vision Model
                response = ollama.chat(
                    model=FINAL_VISION_MODEL,
                    messages=[
                        {
                            'role': 'user',
                            'content': IMAGE_CAPTIONING_LLVM_PROMPT_V3,
                            'images': [base64_image]
                        }
                    ]
                )
                st.session_state.final_vision_model_response = response['message']['content']

                ## LLM Reasoning
                st.session_state.final_reasoning_model_response = get_deepseek_response(FINAL_REASONING_MODEL, st.session_state.user_music_description, st.session_state.final_vision_model_response, USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT_V2)
                
                # Clean the LLM Reasoning Response
                st.session_state.final_reasoning_model_response = extract_tags_content(st.session_state.final_reasoning_model_response,['music_recommendation'])
                # Display the response
                st.write("The description of music we feel would go best with the image: ", st.session_state.final_reasoning_model_response)

                top_summary, top_audio = recommend_with_faiss(st.session_state.final_reasoning_model_response, song_df, rec_model)

                st.session_state.recommendations_list = get_recommendations_list(st.session_state.final_reasoning_model_response)

            st.subheader("📖 Top 5 Songs Based on Summary Reasoning")
            st.table(top_summary.reset_index(drop=True))

            st.subheader("🎧 Top 5 Songs Based on Lyrics + Audio Summary")
            st.table(top_audio.reset_index(drop=True))


        else:
            st.warning("Please upload an image and enter a prompt.")

# Ensure the event loop is run properly
if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function
