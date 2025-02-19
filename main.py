# Workaround for "Buffer" import error in Python 3.9+
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
from configurations import IMAGE_CAPTIONING_LLVM_PROMPT

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def initialize_session_states():
    if "user_music_description" not in st.session_state:
        st.session_state.user_music_description = None
    if "user_uploaded_image" not in st.session_state:
        st.session_state.user_uploaded_image = None

def create_embedding_of_uploaded_image():
    return None

def create_embedding_of_output_vision_model():
    return None

def create_embedding_of_user_input_query():
    return None

async def main():
    # Initialize session state for resume and job link if they don't exist
    initialize_session_states()

    st.title("Music Recommendation using AI")

    st.session_state.user_uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.session_state.user_music_description = st.text_input("Describe what kind of music would you like to add:")

    if st.session_state.user_uploaded_image is not None:
        st.image(st.session_state.user_uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        if st.session_state.user_uploaded_image is not None and st.session_state.user_music_description:
            with st.spinner("Analyzing..."):
                base64_image = encode_image(st.session_state.user_uploaded_image)
                
                response = ollama.chat(
                    model='llava:7b',
                    messages=[
                        {
                            'role': 'user',
                            'content': IMAGE_CAPTIONING_LLVM_PROMPT,
                            'images': [base64_image]
                        }
                    ]
                )
                
                st.write("Analysis Result:")
                st.write(response['message']['content'])
        else:
            st.warning("Please upload an image and enter a prompt.")

# Ensure the event loop is run properly
if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function
