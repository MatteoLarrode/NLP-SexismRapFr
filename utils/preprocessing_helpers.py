# ===============================================
# === Helper functions for data preprocessing ===
# ===============================================
import re
import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
from lingua import Language, LanguageDetectorBuilder
import spacy

# Build the language detector for French
languages = [Language.FRENCH, Language.ENGLISH, Language.SPANISH, Language.GERMAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

# Load the French NLP model for lemmatisation
nlp_fr = spacy.load("fr_core_news_md")

def extract_year(date_str):
    """
    Extract year from various date formats.
    
    This function will be used to determine the year of songs.
    """
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

def is_french(text):
    """
    Determines if the given text is in French.

    This function uses a language detection library to identify the language
    of the input text. If the language is detected as French, it returns True.
    If the detection fails or the language is not French, it returns False.

    Args:
        text (str): The input text to be analyzed.

    Returns:
        bool: True if the text is detected as French, False otherwise.
    """
    try:
        # Using lingua to identify the language
        detected_language = detector.detect_language_of(text)
        return detected_language == Language.FRENCH
    except:
        # If detection fails, return False
        return False
    
def lemmatize(text):
    """
    Lemmatizes the input text using spaCy's French model.
    
    Args:
        text (str): The input text to be lemmatized.
        
    Returns:
        str: The lemmatized text.
    """
    if not isinstance(text, str):
        return ""
        
    doc = nlp_fr(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    
    return lemmatized_text

def clean_tokenise(text, lemmatize=False):
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

    # Use gensim's simple_preprocess for tokenization and cleaning
    tokens = simple_preprocess(text, deacc=False)  # deacc=False prevents the removal of accents

    if lemmatize:
        # Lemmatize the tokens
        tokens = [lemmatize(token) for token in tokens]
    
    return tokens