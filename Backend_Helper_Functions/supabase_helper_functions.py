import pandas as pd
import numpy as np
import json

def prepare_data_music_tracks(df: pd.DataFrame):
    # Function to safely convert values to float and check range
    def convert_to_float(val, min_val=None, max_val=None):
        try:
            val = float(val)
            if min_val is not None and max_val is not None:
                if not (min_val <= val <= max_val):
                    return None  # Return None if out of range
            return round(val, 4)  # Round to 4 decimal places for DECIMAL(5,4)
        except (ValueError, TypeError):
            return None

    # Function to safely convert values to integer and check range
    def convert_to_int(val, min_val=None, max_val=None):
        try:
            val = int(float(val))  # Handles cases like "1.0"
            if min_val is not None and max_val is not None:
                if not (min_val <= val <= max_val):
                    return None  # Return None if out of range
            return val
        except (ValueError, TypeError):
            return None

    # Function to process genres (ensure it's a list)
    def process_genres(val):
        if isinstance(val, list):
            return val
        elif isinstance(val, str):
            # Convert comma-separated string into a list
            return [genre.strip() for genre in val.split(",")]
        else:
            return []

    # Initialize a list to hold all the prepared data for each row
    all_data = []

    # Iterate over each row in the DataFrame
    for idx, row_data in df.iterrows():
        # Prepare the JSON structure for each row
        data = {
            "id": row_data.get("id", None),  # TEXT (string-based ID)
            "artist_name": row_data.get("artist_name", None),  # TEXT (non-nullable)
            "track_name": row_data.get("track_name", None),  # TEXT (non-nullable)
            
            # DECIMAL(5,4) fields with range checks between 0 and 1
            "danceability": convert_to_float(row_data.get("danceability"), 0, 1),
            "energy": convert_to_float(row_data.get("energy"), 0, 1),
            
            # INTEGER fields with specific ranges
            "key": convert_to_int(row_data.get("key"), 0, 11),
            
            # DECIMAL(6,2) for loudness (no specific range in schema)
            "loudness": convert_to_float(row_data.get("loudness")),
            
            # Mode must be an INTEGER and either 0 or 1
            "mode": convert_to_int(row_data.get("mode"), 0, 1),
            
            # DECIMAL(5,4) fields with range checks between 0 and 1
            "speechiness": convert_to_float(row_data.get("speechiness"), 0, 1),
            "acousticness": convert_to_float(row_data.get("acousticness"), 0, 1),
            "instrumentalness": convert_to_float(row_data.get("instrumentalness"), 0, 1),
            "liveness": convert_to_float(row_data.get("liveness"), 0, 1),
            "valence": convert_to_float(row_data.get("valence"), 0, 1),
            
            # DECIMAL(12,4) for duration_ms (no specific range in schema)
            "duration_ms": convert_to_float(row_data.get("duration_ms")),
            
            # Genres must be a list of strings
            "genres": process_genres(row_data.get("genres", []))
        }

        # Append the prepared data for this row to the list
        all_data.append(data)

    # Return the list of dictionaries (ready for insertion into the database)
    return all_data


def audio_features_to_json(df: pd.DataFrame):
    """
    Convert a DataFrame containing audio features to a JSON format suitable for LLM prompting.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns for audio features
    
    Returns:
    str: JSON string representation of the audio features
    """
    # Ensure the DataFrame has the expected columns
    expected_columns = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo'
    ]
    
    # Create a dictionary for each row in the DataFrame
    features_list = []
    for _, row in df.iterrows():
        # Extract features as a dictionary
        features = {}
        for col in expected_columns:
            if col in df.columns:
                features[col] = float(row[col])
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")
        
        features_list.append(features)
    
    # Convert to JSON string with indentation for readability
    json_str = json.dumps(features_list, indent=2)
    return json_str