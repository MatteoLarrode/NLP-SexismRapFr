# ==========================================================
# === Helper functions for data cleaning & preprocessing ===
# ==========================================================
import pandas as pd
import numpy as np
import re
from datetime import datetime
from lingua import Language, LanguageDetectorBuilder 

def is_valid_lyrics(text):
    """
    Checks if the given text is a valid lyrics string.

    A valid lyrics string is defined as:
    - An instance of the `str` type.
    - Having a minimum length of 20 characters (after stripping leading and trailing whitespace).

    Args:
        text (str): The text to validate.

    Returns:
        bool: True if the text is a valid lyrics string, False otherwise.
    """
    if not isinstance(text, str):
        return False
    if len(text.strip()) < 20: 
        return False
    return True

# Build the language detector for French
languages = [Language.FRENCH, Language.ENGLISH, Language.SPANISH, Language.GERMAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

def clean_lyrics(text):
    """
    Cleans the input text by performing the following operations
    1. Removes any text within square brackets (e.g., [Intro], [Chorus]).:
    2. Converts the text to lowercase.
    3. Removes special characters, punctuation, and numbers.
    4. Removes extra whitespace.
    Args:
        text (str): The input text to be cleaned. If the input is not a string, an empty string is returned.
    Returns:
        str: The cleaned text with only lowercase letters and single spaces between words.
    """
    if not isinstance(text, str):
        return ""
    # Remove text between square brackets (e.g., [Intro], [Chorus])
    text = re.sub(r'\[.*?\]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, etc.
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_french(text):
    if not isinstance(text, str) or len(text.strip()) < 20:
        return False

    try:
        # Using lingua to identify the language
        detected_language = detector.detect_language_of(text)
        return detected_language == Language.FRENCH
    except:
        # If detection fails, return False
        return False
    
def extract_year(date_str):
    """Extract year from various date formats."""
    import datetime as dt  # Local import to avoid namespace conflicts
    
    if pd.isna(date_str) or not isinstance(date_str, str):
        return np.nan
    
    # Try different date formats
    formats = ['%Y-%m-%d', '%Y-%m', '%Y', '%d-%m-%Y', '%m-%d-%Y']
    
    for fmt in formats:
        try:
            date_obj = dt.datetime.strptime(date_str, fmt)
            return date_obj.year
        except ValueError:
            continue
    
    # Try to extract just the year if it's in the string
    year_match = re.search(r'(19|20)\d{2}', date_str)
    if year_match:
        return int(year_match.group())
    
    return np.nan