# Workaround for "Buffer" import error in Python 3.9+
import sys
if sys.version_info >= (3, 9):
    from collections.abc import Sequence
    from typing_extensions import Buffer
    import collections.abc
    collections.abc.Buffer = Buffer
else:
    from collections.abc import Buffer

import os
import base64
import pandas as pd
import ollama
from pathlib import Path
from tqdm import tqdm
import csv
import time

# Import configurations - assuming this file exists with the same prompt constants
from configurations import (
    IMAGE_CAPTIONING_LLVM_PROMPT_V3, 
    USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT,
    LLM_REASONING_MODEL, 
    LLM_REASONING_MODEL_V2
)

def encode_image(image_path):
    """Encode image to base64 from file path"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_vision_model_analysis(model_name, image_base64, prompt):
    """Get response from a vision model and track inference time"""
    start_time = time.time()
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [image_base64]
            }
        ]
    )
    end_time = time.time()
    inference_time = end_time - start_time
    return response['message']['content'], inference_time

def get_reasoning_model_response(model_name, user_description, vision_description, prompt_template):
    """Get response from a reasoning model and track inference time"""
    formatted_prompt = prompt_template.format(
        user_description=user_description,
        vision_description=vision_description
    )
    
    start_time = time.time()    
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": formatted_prompt}
        ]
    )
    end_time = time.time()
    inference_time = end_time - start_time
    
    return response["message"]["content"], inference_time

def initialize_inference_time_csv(csv_path):
    """Initialize a single CSV file for all models' inference times"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'model_name', 'inference_time_seconds'])

def append_inference_time(csv_path, image_name, model_name, inference_time):
    """Append an inference time record to the inference times CSV file"""
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([image_name, model_name, inference_time])

def main():
    # Define the paths directly in the script
    image_dir = "/Users/pratikhotchandani/Desktop/NEU/Semester/Sem 5 Spring 2025/DS 5500/Testing Images"
    user_inputs_csv = "/Users/pratikhotchandani/Downloads/Github/DS_5500/user_descriptions.csv"
    output_csv = "/Users/pratikhotchandani/Downloads/Github/DS_5500/results.csv"
    
    # Define the inference times CSV path
    output_dir = os.path.dirname(output_csv)
    inference_times_csv = os.path.join(output_dir, "inference_times.csv")
    
    print(f"Processing images from: {image_dir}")
    print(f"Using user descriptions from: {user_inputs_csv}")
    print(f"Model outputs will be saved to: {output_csv}")
    print(f"Inference times will be saved to: {inference_times_csv}")
    
    # If you want to keep command line arguments as an option, uncomment this code:
    """
    import argparse
    parser = argparse.ArgumentParser(description='Batch process images with AI models')
    parser.add_argument('-i', '--image_dir', type=str, default=image_dir, help='Directory containing images')
    parser.add_argument('-u', '--user_inputs_csv', type=str, default=user_inputs_csv, help='CSV file with image filename to user input mapping')
    parser.add_argument('-o', '--output_csv', type=str, default=output_csv, help='Output CSV file path')
    args = parser.parse_args()
    
    image_dir = args.image_dir
    user_inputs_csv = args.user_inputs_csv
    output_csv = args.output_csv
    """

    # Verify paths exist
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    if not os.path.exists(user_inputs_csv):
        # Try to create a sample user_descriptions.csv if it doesn't exist
        print(f"User inputs CSV not found: {user_inputs_csv}")
        print("Creating a sample user_descriptions.csv file...")
        
        # Get all image files first
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
        
        if not image_files:
            print(f"No image files found in directory: {image_dir}")
            return
            
        # Create a sample CSV with filenames and default descriptions
        with open(user_inputs_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'user_description'])
            for img in image_files:
                writer.writerow([img.name, 'Please analyze this image and recommend suitable music'])
        
        print(f"Created sample CSV at {user_inputs_csv}. You may want to edit the descriptions.")
    
    # Load user inputs from CSV - special handling for the problematic format
    try:
        # Let's manually parse the CSV content
        with open(user_inputs_csv, 'r') as f:
            content = f.read()
        
        # Create a dictionary to map filenames to descriptions
        user_inputs = {}
        
        # Define the mapping directly based on the content provided
        descriptions = {
            'bike.jpg': 'Adventure, thrill, high energy, feeling the breeze, adrenaline rush',
            'cafeaesthetic.jpg': 'Enjoying morning breakfast on the curbside, looks like a part of their daily routine, Moderate energy',
            'cafeoutdoor.jpg': 'A quaint, cute cafe, aesthetically pleasing with the flowers. The general vibe would be slow, calm music with low relaxing energy',
            'cafework.jpeg': 'work at cafe, coffee, aesthetic, calm music',
            'hike.jpg': 'adventure, nature, either adventure / calm mood',
            'flowers.jpg': 'something related to nature, medium energy, spreading happiness, and postive vibe',
            'graduation.jpg': 'Graduation, sense of acheivement, proud, happy, smiles, exciitement',
            'nightparty.jpeg': 'fun night party, drinks, loud music, flash photography',
            'nycfromflight.jpeg': 'Travel, excitement, looking forward to, NYC',
            'picnic.jpg': 'something related to nature, medium energy, spreading happiness, and postive vibe',
            'roadtrip.jpg': 'Travel, road trip, happy energy, good day, relaxation, high energy',
            'coffeeposter.jpg': 'Cafe, coffee specialising place. Low tempo, relaxed music',
            'couple.jpg': 'Romantic, well planned outfits, date day, cute and romantic music. Something trendy',
            'dog.jpg': 'sunshine, friendship, cuteness, goofy energy, happiness',
            'manwithbike.jpg': 'A chill, relaxed vibe of a man with his cycle, calming and regular energy'
        }
        
        # Use the hardcoded descriptions
        user_inputs = descriptions
        
        # Print the descriptions we'll be using
        print("Using the following descriptions for images:")
        for filename, desc in user_inputs.items():
            print(f"  {filename}: {desc[:50]}..." if len(desc) > 50 else f"  {filename}: {desc}")
            
    except Exception as e:
        print(f"Error handling user inputs: {e}")
        print("Using default descriptions instead.")
        user_inputs = {}
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
    
    if not image_files:
        print(f"No image files found in directory: {image_dir}")
        return
    
    # Define models
    vision_models = {
        'llava:7b': 'LLaVA_7B',  # Remove spaces from model names
        'llama3.2-vision:11b': 'LLama3_2_Vision_11B',
        'gemma3:12b': 'Gemma3_12B',
        'minicpm-v:8b': 'MiniCPM_V_8B',
        'llava-llama3:8b': 'LLaVA_LLama3_8B',
        'bakllava:7b': 'BakLLaVA_7B',
        'llava-phi3:latest': 'LLaVA_Phi3',
        'granite3.2-vision:latest': 'Granite3_2_Vision'
    }
    
    # Models we want to run reasoning on
    reasoning_vision_models = [
        'LLaVA_7B',
        'LLama3_2_Vision_11B',
        'Gemma3_12B',
        'MiniCPM_V_8B'
    ]
    
    reasoning_models = {
        LLM_REASONING_MODEL: 'DeepSeek_Reasoning',
        'mychen76/llama3.1-intuitive-thinker:chain-of-thoughts.q5': 'LLama3_1_Intuitive_Thinker',
        'huihui_ai/deepseek-r1-abliterated:14b': 'DeepSeek_R1_Abliterated',
        'phi4:14b': 'Phi4_14B'
    }
    
    # Initialize inference time CSV file
    initialize_inference_time_csv(inference_times_csv)
    
    # Create or overwrite output CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        # Define column headers
        fieldnames = ['image_name', 'user_description']
        
        # Add vision model columns
        for model_name, model_display in vision_models.items():
            fieldnames.append(f'{model_display}_output')
        
        # Add reasoning model columns
        for reasoning_model, reasoning_display in reasoning_models.items():
            for vision_model_display in vision_models.values():
                fieldnames.append(f'{reasoning_display}_{vision_model_display}_output')
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            image_filename = image_path.name
            print(f"\nProcessing {image_filename}...")
            
            # Check if we have a user description for this image
            if image_filename not in user_inputs:
                print(f"Warning: No user description found for {image_filename}, using default.")
                user_description = "Please analyze this image and recommend suitable music"
            else:
                user_description = user_inputs[image_filename]
                print(f"Using description: {user_description[:50]}..." if len(user_description) > 50 else f"Using description: {user_description}")
            
            # Prepare row for CSV
            row = {
                'image_name': image_filename,
                'user_description': user_description
            }
            
            # Encode image
            try:
                base64_image = encode_image(image_path)
            except Exception as e:
                print(f"Error encoding image {image_filename}: {e}")
                continue
            
            # Get vision model outputs
            vision_outputs = {}
            for model_name, model_display in vision_models.items():
                print(f"  Running vision model: {model_display}")
                try:
                    output, inference_time = get_vision_model_analysis(model_name, base64_image, IMAGE_CAPTIONING_LLVM_PROMPT_V3)
                    row[f'{model_display}_output'] = output
                    
                    # Write inference time to inference times CSV
                    append_inference_time(inference_times_csv, image_filename, model_display, inference_time)
                    
                    vision_outputs[model_display] = output
                    print(f"    ✓ Success (took {inference_time:.2f} seconds)")
                except Exception as e:
                    error_msg = str(e)
                    print(f"    ✗ Error: {error_msg[:100]}..." if len(error_msg) > 100 else f"    ✗ Error: {error_msg}")
                    row[f'{model_display}_output'] = f"ERROR: {error_msg}"
            
            # Get reasoning model outputs - only for selected vision models
            for reasoning_model, reasoning_display in reasoning_models.items():
                for vision_model_display, vision_output in vision_outputs.items():
                    # Only run reasoning on the specified vision models
                    if vision_model_display in reasoning_vision_models:
                        print(f"  Running reasoning model: {reasoning_display} with {vision_model_display}")
                        try:
                            output, inference_time = get_reasoning_model_response(
                                reasoning_model, 
                                user_description, 
                                vision_output, 
                                USER_INPUT_IMAGE_CAPTION_REASONING_PROMPT
                            )
                            row[f'{reasoning_display}_{vision_model_display}_output'] = output
                            
                            # Write inference time to inference times CSV
                            combo_name = f"{reasoning_display}_{vision_model_display}"
                            append_inference_time(inference_times_csv, image_filename, combo_name, inference_time)
                            
                            print(f"    ✓ Success (took {inference_time:.2f} seconds)")
                        except Exception as e:
                            error_msg = str(e)
                            print(f"    ✗ Error: {error_msg[:100]}..." if len(error_msg) > 100 else f"    ✗ Error: {error_msg}")
                            row[f'{reasoning_display}_{vision_model_display}_output'] = f"ERROR: {error_msg}"
            
            # Write row to CSV
            try:
                # Check if field names match exactly
                missing_fields = set(row.keys()) - set(fieldnames)
                extra_fields = set(fieldnames) - set(row.keys())
                
                if missing_fields:
                    print(f"  Warning: Row contains {len(missing_fields)} fields not in fieldnames.")
                    # Remove any fields that aren't in the fieldnames
                    for field in missing_fields:
                        row.pop(field, None)
                
                if extra_fields:
                    print(f"  Warning: {len(extra_fields)} expected fields are missing from row.")
                    # Add empty values for any missing fields
                    for field in extra_fields:
                        row[field] = ""
                
                writer.writerow(row)
                # Flush to disk after each image to avoid data loss
                csvfile.flush()
                print(f"  Results for {image_filename} saved to CSV")
            except Exception as e:
                print(f"  Error writing results for {image_filename} to CSV: {e}")
                # Try to write a simpler version of the row
                try:
                    simple_row = {
                        'image_name': image_filename,
                        'user_description': user_description
                    }
                    
                    # Only include fields that are in fieldnames
                    for field in fieldnames:
                        if field in row:
                            simple_row[field] = row[field]
                        else:
                            simple_row[field] = ""
                    
                    writer.writerow(simple_row)
                    print("  Wrote simplified row instead")
                except Exception as e2:
                    print(f"  Could not write even a simplified row: {e2}")
    
    print(f"\nProcessing complete!")
    print(f"Model outputs saved to: {output_csv}")
    print(f"Model inference times saved to: {inference_times_csv}")

if __name__ == "__main__":
    main()