# ===============================================
# === Helper functions for data preprocessing ===
# ===============================================
import json
import pickle
import requests
import re
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
from lingua import Language, LanguageDetectorBuilder
import spacy
from tqdm import tqdm

# Download NLTK resources
nltk.download('punkt', quiet=True)

# Build the language detector for French
languages = [Language.FRENCH, Language.ENGLISH, Language.SPANISH, Language.GERMAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

# In the terminal, run: python -m spacy download fr_core_news_md
# Load French spaCy model if available
try:
    nlp_fr = spacy.load("fr_core_news_md")
    HAS_FRENCH_LEMMATIZER = True
except (ImportError, OSError):
    HAS_FRENCH_LEMMATIZER = False
    print("Warning: French spaCy model not available.")
    print("Install with: python -m spacy download fr_core_news_md")

def download_lyrics():
    """
    Downloads French rap lyrics data from GitHub and returns them as a numpy array.
    
    Returns:
        np.ndarray: A numpy array of lyrics strings.
    """
    # URLs for the data
    old_songs_url = "https://raw.githubusercontent.com/ljz112/CLResearch/refs/heads/main/dataEntries/frenchDataOldSongs.json"
    new_songs_url = "https://raw.githubusercontent.com/ljz112/CLResearch/refs/heads/main/dataEntries/frenchDataNew.json"
    
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
    new_songs_df = pd.DataFrame(data_new['allSongs'])
    
    # Combine the DataFrames
    all_songs_df = pd.concat([old_songs_df, new_songs_df], ignore_index=True)
    
    # Only keep the lyrics, and make it a numpy array
    all_songs_lyrics = all_songs_df['lyrics'].to_numpy()
    
    print(f"Downloaded {len(all_songs_lyrics)} songs")

    # Save lyrics to a pickle file
    with open("data/french_rap_lyrics_raw.pkl", "wb") as f:
        pickle.dump(all_songs_lyrics, f)

    print("Lyrics saved to data/french_rap_lyrics_raw.pkl")

    return all_songs_lyrics

# def extract_year(date_str):
#     """
#     Extract year from various date formats.
    
#     This function will be used to determine the year of songs.
#     """
#     import datetime as dt  # Local import to avoid namespace conflicts
    
#     if pd.isna(date_str) or not isinstance(date_str, str):
#         return np.nan
    
#     # Try different date formats
#     formats = ['%Y-%m-%d', '%Y-%m', '%Y', '%d-%m-%Y', '%m-%d-%Y']
    
#     for fmt in formats:
#         try:
#             date_obj = dt.datetime.strptime(date_str, fmt)
#             return date_obj.year
#         except ValueError:
#             continue
    
#     # Try to extract just the year if it's in the string
#     year_match = re.search(r'(19|20)\d{2}', date_str)
#     if year_match:
#         return int(year_match.group())
    
#     return np.nan

def is_french(text):
    """
    Determines if the given text is in French.

    Args:
        text (str): The input text to be analyzed.

    Returns:
        bool: True if the text is detected as French, False otherwise.
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return False
        
    try:
        # Using lingua to identify the language
        detected_language = detector.detect_language_of(text)
        return detected_language == Language.FRENCH
    except:
        # If detection fails, return False
        return False
    
def lemmatize_text(text):
    """
    Lemmatizes a single text using spaCy's French model.
    
    Args:
        text (str): The input text to be lemmatized.
        
    Returns:
        list: List of lemmatized tokens.
    """
    if not HAS_FRENCH_LEMMATIZER:
        return []
        
    if not isinstance(text, str):
        return []
        
    doc = nlp_fr(text.lower())
    # Extract lemmas, filter out punctuation and short words
    tokens = [token.lemma_ for token in doc 
             if token.is_alpha and len(token.lemma_) > 2]
    return tokens

def clean_text(text, lemmatize=False):
    """
    Cleans the input text to fit the needs of gensim models.
    
    Args:
        text (str): The input text to be cleaned.
        lemmatize (bool): Whether to lemmatize the text or not.
        
    Returns:
        list: A list of cleaned, tokenized words.
    """
    if not isinstance(text, str):
        return []
        
    if len(text.strip()) < 20:
        return []
        
    # Remove text between square brackets (e.g., [Intro], [Chorus])
    text = re.sub(r'\[.*?\]', '', text)
    
    if lemmatize and HAS_FRENCH_LEMMATIZER:
        return lemmatize_text(text)
    else:
        # Use gensim's simple_preprocess for tokenization and cleaning
        tokens = simple_preprocess(text, deacc=False)  # deacc=False preserves accents
        return tokens
    
def preprocess_corpus(texts, lemmatize=False, save_path=None):
    """
    WRAPPER: Preprocesses a numpy array or list of texts to create a corpus for word2vec training.
    
    Args:
        texts (np.ndarray or list): Array or list of text strings to preprocess.
        lemmatize (bool): Whether to lemmatize the texts.
        
    Returns:
        list: List of tokenized texts for training.
    """
    print(f"Preprocessing {len(texts)} songs...")
    # Handle numpy array
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    
    corpus = []
    
    for text in tqdm(texts, desc="Processing texts"):
        # Skip non-French texts
        if not is_french(text):
            continue
            
        tokens = clean_text(text, lemmatize)
        if tokens:  # Only add non-empty token lists
            corpus.append(tokens)
    
    # Remove empty lists
    corpus = [tokens for tokens in corpus if tokens]

    # Remove duplicates while preserving order
    seen = set()
    corpus = [tokens for tokens in corpus if tuple(tokens) not in seen and not seen.add(tuple(tokens))]
    
    print(f"Preprocessing finished! {len(corpus)} songs valid.")

    # Save the corpus to a pickle file if a path is provided
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(corpus, f)
        print(f"Corpus saved to {save_path}")
            
    return