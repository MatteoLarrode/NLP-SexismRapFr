"""
Data setup script for French rap lyrics analysis.
Downloads, processes, and saves French rap song data for later analysis.
"""
import json
import pandas as pd
import requests
import os
import argparse
from tqdm import tqdm

# Import helper functions
from utils.cleaning_helpers import (
    is_valid_lyrics,
    clean_lyrics,
    is_french,
    extract_year,
)

def main():
    """Main function to download and process data."""
    parser = argparse.ArgumentParser(description='Download and process French rap lyrics data')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading data if already downloaded')
    args = parser.parse_args()
    
    # URLs for the data
    old_songs_url = "https://raw.githubusercontent.com/ljz112/CLResearch/refs/heads/main/dataEntries/frenchDataOldSongs.json"
    new_songs_url = "https://raw.githubusercontent.com/ljz112/CLResearch/refs/heads/main/dataEntries/frenchDataNew.json"
    
    if not args.skip_download or not (os.path.exists("data/french_rap_songs.csv") and os.path.exists("data/french_rap_artists.csv")):
        # Download the data
        print("Downloading old songs data...")
        old_songs_response = requests.get(old_songs_url)
        data_old = json.loads(old_songs_response.text)
        
        print("Downloading new songs data...")
        new_songs_response = requests.get(new_songs_url)
        data_new = json.loads(new_songs_response.text)
        
        print("Data downloaded successfully.")
        
        # Create DataFrames
        old_songs_df = pd.DataFrame(data_old['allSongs'])
        old_artists_df = pd.DataFrame(data_old['allArtists'])
        
        new_songs_df = pd.DataFrame(data_new['allSongs'])
        new_artists_df = pd.DataFrame(data_new['allArtists'])
        
        # Combine the DataFrames
        all_songs_df = pd.concat([old_songs_df, new_songs_df], ignore_index=True)
        all_artists_df = pd.concat([old_artists_df, new_artists_df], ignore_index=True)
        
        # Save the raw DataFrames to CSV files
        all_songs_df.to_csv("data/french_rap_songs.csv", index=False)
        all_artists_df.to_csv("data/french_rap_artists.csv", index=False)
        print("Raw data saved to CSV files.")
    else:
        # Load existing data
        print("Loading existing data from CSV files...")
        all_songs_df = pd.read_csv("data/french_rap_songs.csv")
        all_artists_df = pd.read_csv("data/french_rap_artists.csv")
    
    # Process songs dataframe
    print("Processing songs data...")
    songs_df = all_songs_df.copy()
    
    # Filter out songs with invalid lyrics
    print(f"Total songs before filtering: {len(songs_df)}")
    songs_df = songs_df[songs_df['lyrics'].apply(is_valid_lyrics)]
    print(f"Songs after filtering invalid lyrics: {len(songs_df)}")
    
    # Clean lyrics
    print("Cleaning lyrics...")
    songs_df['cleaned_lyrics'] = songs_df['lyrics'].apply(clean_lyrics)
    
    # Filter for French lyrics only
    print("Detecting French lyrics...")
    # tqdm for progress bar
    tqdm.pandas(desc="Language detection")
    songs_df['is_french'] = songs_df['lyrics'].progress_apply(is_french)
    songs_df = songs_df[songs_df['is_french']]
    print(f"Songs after filtering for French language: {len(songs_df)}")
    
    # Extract year and create decade
    print("Processing dates...")
    songs_df['year'] = songs_df['releaseDate'].apply(extract_year)
    songs_df = songs_df.dropna(subset=['year'])
    songs_df['decade'] = ((songs_df['year'] // 10) * 10).astype(int).astype(str) + 's'
    
    # Save the processed dataframe
    print("Saving processed data...")
    songs_df.to_csv("data/processed_french_rap_songs.csv", index=False)
    
    # Print summary statistics
    print("\nData Processing Complete!")
    print(f"Number of songs: {len(songs_df)}")
    if not songs_df.empty:
        print(f"Year range: {int(songs_df['year'].min())} to {int(songs_df['year'].max())}")
        print(f"Number of songs by decade:")
        print(songs_df['decade'].value_counts().sort_index())
    
    print("\nProcessed data saved to 'data/processed_french_rap_songs.csv'")


if __name__ == "__main__":
    main()