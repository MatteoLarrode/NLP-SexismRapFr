# =========================================================
# === Helper functions for the word embeddings analysis ===
# =========================================================
import os
from gensim.models import Word2Vec

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