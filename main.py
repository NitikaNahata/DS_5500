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
from configurations import IMAGE_CAPTIONING_LLVM_PROMPT, IMAGE_CAPTIONING_LLVM_PROMPT_V2, IMAGE_CAPTIONING_LLVM_PROMPT_V3, USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT, LLM_REASONING_MODEL, LLM_REASONING_MODEL_V2

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def initialize_session_states():
    if "user_music_description" not in st.session_state:
        st.session_state.user_music_description = None
    if "user_uploaded_image" not in st.session_state:
        st.session_state.user_uploaded_image = None
    if "llava_response" not in st.session_state:
        st.session_state.llava_response = None
    if "llama_response" not in st.session_state:
        st.session_state.llama_response = None
    if "gemma_response" not in st.session_state:
        st.session_state.gemma_response = None

# Function to extract thinking and content from response
def parse_response(response):
    message_content = response["message"]["content"]
    thinking_content = message_content.split("<think>")[1].split("</think>")[0].strip()
    response_content = message_content.split("</think>")[1].strip()
    return thinking_content, response_content

def create_embedding_of_uploaded_image():
    return None

def create_embedding_of_output_vision_model():
    return None

def create_embedding_of_user_input_query():
    return None

def get_deepseek_response(model_name, user_description, vision_description, prompt_template):
    # Format the prompt with user inputs
    # Format the prompt with user inputs
    formatted_prompt = prompt_template.format(
        user_description=user_description,
        vision_description=vision_description
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
                
                ## LLaVA Model
                response = ollama.chat(
                    model='llava:7b',
                    messages=[
                        {
                            'role': 'user',
                            'content': IMAGE_CAPTIONING_LLVM_PROMPT_V3,
                            'images': [base64_image]
                        }
                    ]
                )         
                st.write("Analysis Result LLaVA 7B: ")
                st.write(response['message']['content'])
                st.session_state.llava_response = response['message']['content']

                ## LLama 3.2 Vision 11B Model
                response = ollama.chat(
                    model='llama3.2-vision:11b',
                    messages=[
                        {
                            'role': 'user',
                            'content': IMAGE_CAPTIONING_LLVM_PROMPT_V3,
                            'images': [base64_image]
                        }
                    ]
                )               
                st.write("Analysis Result llama3.2-vision:11b:")
                st.write(response['message']['content'])
                st.session_state.llama_response = response['message']['content']

                ## Gemma 3 Vision 27B Model
                #response = ollama.chat(
                #    model='gemma3:27b',
                #    messages=[
                #        {
                #            'role': 'user',
                #            'content': IMAGE_CAPTIONING_LLVM_PROMPT_V3,
                #            'images': [base64_image]
                #        }
                #    ]
                #)               
                #st.write("Analysis Result gemma3:27b:")
                #st.write(response['message']['content'])
                #st.session_state.gemma_response = response['message']['content']

                st.write("REASONING WITH DEEPSEEK !!!! ")

                ## LLM Reasoning
                st.write("\nGenerating Reasoning response for LLaVA...")
                response = get_deepseek_response(LLM_REASONING_MODEL, st.session_state.user_music_description, st.session_state.llava_response, USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT)
                # Display the response
                with st.expander("LLaVA resoning: "):
                    st.write(response)

                st.write("\nGenerating Reasoning response for LLama...")
                response = get_deepseek_response(LLM_REASONING_MODEL, st.session_state.user_music_description, st.session_state.llama_response, USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT)
                # Display the response
                with st.expander("LLama Reasoning: "):
                    st.write(response)

                """
                st.write("\nGenerating Reasoning response for Gemma...")
                response = get_deepseek_response(LLM_REASONING_MODEL, st.session_state.user_music_description, st.session_state.gemma_response, USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT)
                # Display the response
                with st.expander("Gemma Reasoning: "):
                    st.write(st.session_state.response)

                st.write("REASONING WITH QWENN !!!! ")

                ## LLM Reasoning
                st.write("\nGenerating Reasoning response for LLaVA...")
                response = get_deepseek_response(LLM_REASONING_MODEL_V2, st.session_state.user_music_description, st.session_state.llava_response, USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT)
                # Display the response
                with st.expander("LLaVA resoning: "):
                    st.write(response)

                st.write("\nGenerating Reasoning response for LLama...")
                response = get_deepseek_response(LLM_REASONING_MODEL_V2, st.session_state.user_music_description, st.session_state.llama_response, USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT)
                # Display the response
                with st.expander("LLama Reasoning: "):
                    st.write(response)

                st.write("\nGenerating Reasoning response for Gemma...")
                response = get_deepseek_response(LLM_REASONING_MODEL_V2, st.session_state.user_music_description, st.session_state.gemma_response, USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT)
                # Display the response
                with st.expander("Gemma Reasoning: "):
                    st.write(st.session_state.response)
                """



        else:
            st.warning("Please upload an image and enter a prompt.")

# Ensure the event loop is run properly
if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function
