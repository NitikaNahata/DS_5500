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

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

st.title("Image Analysis with LLaVA")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
prompt = st.text_input("Enter your prompt:")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if st.button("Analyze"):
    if uploaded_file is not None and prompt:
        with st.spinner("Analyzing..."):
            base64_image = encode_image(uploaded_file)
            
            response = ollama.chat(
                model='llava:7b',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [base64_image]
                    }
                ]
            )
            
            st.write("Analysis Result:")
            st.write(response['message']['content'])
    else:
        st.warning("Please upload an image and enter a prompt.")