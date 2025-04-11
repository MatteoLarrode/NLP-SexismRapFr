# ==============================================
# === Helper functions for Word2Vec training ===
# ==============================================
import os
import json
import pickle
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

def train_word2vec(
    corpus,
    output_dir='models',
    lemmatized=False,
    vector_size=100,
    window=10,
    min_count=5,
    sg=1,  # 1 for skip-gram, 0 for CBOW
    epochs=5,
    workers=4,
    seed=42
):
    """
    Train a word2vec model on the given corpus.
    
    Args:
        corpus (list): List of tokenized texts.
        output_dir (str): Base directory to save the model and related files.
        lemmatized (bool): Whether the corpus was lemmatized.
        vector_size (int): Dimensionality of the word vectors.
        window (int): Maximum distance between current and predicted word.
        min_count (int): Minimum count of words to consider.
        sg (int): Training algorithm: 1 for skip-gram, 0 for CBOW.
        epochs (int): Number of training epochs.
        workers (int): Number of worker threads.
        seed (int): Random seed for reproducibility.
        
    Returns:
        model: Trained Word2Vec model.
    """
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate standardized model name
    algorithm = "skipgram" if sg == 1 else "cbow"
    lemma_status = "lemma" if lemmatized else "non_lemma"
    model_name = f"frRap_{lemma_status}_{algorithm}_{vector_size}_{min_count}"
    
    # Create model-specific directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define paths for model files
    model_path = os.path.join(model_dir, "model")
    vocab_path = os.path.join(model_dir, "vocab.pkl")
    vectors_path = os.path.join(model_dir, "vectors.npy")
    
    # Train the model
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=sg,
        seed=seed
    )
    
    # Save the model and related files
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save vocabulary dictionary
    vocab_dict = {word: index for index, word in enumerate(model.wv.index_to_key)}
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_dict, f)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Save vectors in numpy format
    np.save(vectors_path, model.wv.vectors)
    print(f"Vectors saved to {vectors_path}")
    
    # Save model info for reference
    info = {
        'name': model_name,
        'lemmatized': lemmatized,
        'algorithm': algorithm,
        'vector_size': vector_size,
        'window': window,
        'min_count': min_count,
        'epochs': epochs,
        'vocab_size': len(model.wv.index_to_key),
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(model_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"All model files saved in: {model_dir}")
    return model