# =========================================================
# === Helper functions for the word embeddings analysis ===
# =========================================================
import os
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import List, Dict, Tuple

# ===== Handling Word2Vec Models =====
def load_models(base_dir='models'):
    """Load all Word2Vec models from the specified directory."""
    models = {}
    
    # List all subdirectories
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Look for model files in this directory
            for file in os.listdir(subdir_path):
                if file.endswith('_model'):
                    model_path = os.path.join(subdir_path, file)
                    model_name = subdir
                    try:
                        models[model_name] = Word2Vec.load(model_path)
                        print(f"Loaded model: {model_name}")
                    except Exception as e:
                        print(f"Error loading {model_path}: {e}")
    
    return models

def most_similar_words(model, word, topn=10):
    """Find the most similar words to a given word in a model."""
    if word not in model.wv:
        return f"'{word}' not in vocabulary"
    
    return model.wv.most_similar(word, topn=topn)

def compare_word_across_models(models, word, topn=10):
    """Compare similar words for a target word across all models."""
    results = {}
    
    for model_name, model in models.items():
        if word in model.wv:
            results[model_name] = model.wv.most_similar(word, topn=topn)
        else:
            print(f"'{word}' not in vocabulary for model {model_name}")
    
    return results

def compare_word_pairs(models, word_pairs):
    """Compare similarity between word pairs across all models."""
    results = {}
    
    for model_name, model in models.items():
        model_results = {}
        for word1, word2 in word_pairs:
            if word1 in model.wv and word2 in model.wv:
                similarity = model.wv.similarity(word1, word2)
                model_results[f"{word1}-{word2}"] = similarity
            else:
                if word1 not in model.wv:
                    print(f"'{word1}' not in vocabulary for model {model_name}")
                if word2 not in model.wv:
                    print(f"'{word2}' not in vocabulary for model {model_name}")
        
        results[model_name] = model_results
    
    return results

# ===== WEAT experiment =====
# Define the attribute and target word lists for French rap
attribute_words = {
    'M': ['homme', 'mec', 'gars', 'frère', 'il', 'lui', 'son', 'fils', 'père', 'oncle', 'grand-père', 'mâle', 'king', 'bro', 'kho'],
    'F': ['femme', 'meuf', 'fille', 'sœur', 'elle', 'sa', 'fille', 'mère', 'tante', 'grand-mère', 'gazelle', 'go', 'miss', 'bae']
}

target_words = {
    'B1_career_family': {
        'X_career': ['business', 'patron', 'patronne', 'money', 'travail', 'boss', 'cash', 'hustle', 'bureau', 'carrière'],
        'Y_family': ['foyer', 'parents', 'maison', 'enfants', 'famille', 'mariage', 'domestique']
    },
    'B2_mathsci_arts': {
        'X_mathsci': ['calcul', 'logique', 'science', 'chiffres', 'physique', 'maths', 'chimie'],
        'Y_arts': ['poésie', 'art', 'danse', 'littérature', 'chanson', 'peinture']
    },
    'B3_intel_appearance': {
        'X_intel': ['brillant','brillants', 'intelligent', 'intelligente', 'stratège', 'cerveau','sage', 'lucide', 'génie'],
        'Y_appearance': ['beau', 'belle', 'mince', 'moche', 'laid', 'laide', 'joli', 'jolie', 'maigre', 'gros', 'grosse', 'corps']
    },
    'B4_strength_weakness': {
        'X_strength': ['confiant','confiante', 'puissant', 'puissante', 'force', 'dominat', 'dominante', 'fort', 'forte'],
        'Y_weakness': ['faible', 'fragile', 'timide' , 'doux' ,'douce', 'sensible', 'soumis', 'soumise', 'peur', 'vulnérable']
    },
    'B5_status_love': {
        'X_status': ['oseille', 'thune', 'francs', 'euros', 'dollars', 'bijoux', 'marques', 'luxe', 'rolex', 'chaine', 'fric'],
        'Y_love': ['amour', 'sentiments', 'cœur', 'passion', 'fidèle', 'romantique', 'relation', 'aimer', 'émotions', 'attachement']
    }
}