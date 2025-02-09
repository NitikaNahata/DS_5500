import pandas as pd

def prepare_data_music_tracks(df: pd.DataFrame):
    # Initialize a list to hold all the prepared data for each row
    all_data = []

    # Iterate over each row in the DataFrame and corresponding embeddings
    for idx, row_data in df.iterrows():  # Unpack the index and row data
        # Prepare the JSON structure for each row, including the embeddings
        data = {
            "id": row_data.get("id", None),   # Extract id
            "artist_name": row_data.get("artist_name", None),  # Handle artist_name for summary
            "track_name": row_data.get("track_name", None),   # Extract the track_name from the DataFrame
            "danceability": row_data.get("danceability", None),
            "energy": row_data.get("energy", None),
            "key": row_data.get("key", None),
            "loudness": row_data.get("loudness", None),
            "mode": row_data.get("mode", None),
            "speechiness": row_data.get("speechiness", None),
            "acousticness": row_data.get("acousticness", None),
            "instrumentalness": row_data.get("instrumentalness", None),
            "liveness": row_data.get("liveness", None),
            "valence": row_data.get("valence", None),
            "duration_ms": row_data.get("duration_ms", None),
            "genres": row_data.get("genres", [])  # Ensure genres is an array
        }

        # Append the prepared data for this row to the list
        all_data.append(data)

    # Return the list of dictionaries (ready for insertion into the database)
    return all_data
