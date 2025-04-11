# ==============================================
# === Helper functions for Word2Vec training ===
# ==============================================
import os
import pickle
import numpy as np
from gensim.models import Word2Vec

def train_word2vec(
    corpus,
    output_path,
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
        output_path (str): Path to save the model and related files.
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
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
    
    # Save the model
    model.save(output_path)
    
    # Save vocabulary dictionary
    vocab_dict = {word: index for index, word in enumerate(model.wv.index_to_key)}
    with open(f'{output_path}-vocab.pkl', 'wb') as f:
        pickle.dump(vocab_dict, f)
    
    # Save vectors in numpy format
    np.save(f'{output_path}-vectors.npy', model.wv.vectors)
    
    return model