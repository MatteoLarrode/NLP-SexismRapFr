
"""
Word2Vec trainer for French rap lyrics.

This script includes code for the preprocessing of lyrics
and the training Word2Vec models with configurable parameters.
"""
import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import logging
from tqdm import tqdm

# === Configure logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Text preprocessing helper functions ===
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

def clean_lyrics(text):
    """
    Cleans the input text to fit the needs of gensim models by performing the following operations:
    1. Removes any text within square brackets (e.g., [Intro], [Chorus]).
    2. Converts the text to lowercase.
    3. Removes special characters, punctuation, and numbers.
    4. Tokenizes the text into a list of lowercase words using gensim's simple_preprocess.
    
    Args:
        text (str): The input text to be cleaned. If the input is not a string, an empty list is returned.
    
    Returns:
        list: A list of cleaned, tokenized words suitable for gensim models.
    """
    if not isinstance(text, str):
        return []
        
    # Remove text between square brackets (e.g., [Intro], [Chorus])
    text = re.sub(r'\[.*?\]', '', text)
    
    # Use gensim's simple_preprocess for tokenization and cleaning
    tokens = simple_preprocess(text, deacc=False)  # deacc=False prevents the removal of accents
    
    return tokens

def extract_year(date_str):
    """Extract year from various date formats."""
    if pd.isna(date_str) or not isinstance(date_str, str):
        return np.nan
        
    # Try different date formats
    import datetime as dt  # Local import
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

def assign_decade(year):
    """Assign a decade category to a year."""
    if pd.isna(year):
        return np.nan
        
    year = int(year)
    if 1990 <= year < 2010:
        return "1990s-2000s"
    else:
        return f"{(year // 10) * 10}s"
