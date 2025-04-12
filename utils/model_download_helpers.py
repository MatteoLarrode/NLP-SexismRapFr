# =============================================================
# === Helper functions to download off-the-shelf embeddings ===
# =============================================================
import os
import requests
import json
import tempfile
from gensim.models import KeyedVectors, Word2Vec
from tqdm import tqdm
import urllib.parse

def download_word2vec_model(url, base_dir='models'):
    """
    Download a word2vec model from a URL, convert it to the required format,
    and store it efficiently without keeping the original binary file.
    
    Args:
        url (str): URL to download the model from
        base_dir (str): Base directory to save models to
        
    Returns:
        str: Path to the saved model directory
    """
    # Extract the filename from the URL
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    
    # Use filename without extension as the model name
    model_name = os.path.splitext(filename)[0]
    
    # Create target directory
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a temporary file for downloading
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Download the file
        print(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(temp_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"Download complete. Loading model into memory...")
        
        # Load the binary model directly from the temp file
        kv_model = KeyedVectors.load_word2vec_format(temp_path, binary=True, unicode_errors="ignore")
        
        # Convert KeyedVectors to Word2Vec model
        word2vec_model = Word2Vec(vector_size=kv_model.vector_size)
        word2vec_model.wv = kv_model
        
        # Save model in the required format
        model_path = os.path.join(model_dir, "model")
        print(f"Saving model to {model_path}...")
        word2vec_model.save(model_path)
        
        # Extract basic metadata
        vocab_size = len(kv_model.index_to_key)
        vector_size = kv_model.vector_size
        
        # Calculate the total number of tokens if count attribute exists
        try:
            total_tokens = sum(kv_model.get_vecattr(word, "count") for word in kv_model.index_to_key)
        except:
            total_tokens = None
        
        # Create info.json
        info = {
            'source_url': url,
            'filename': filename,
            'vocabulary_size': vocab_size,
            'vector_size': vector_size,
            'total_tokens': total_tokens
        }
        
        # Try to parse additional info from the filename
        parts = model_name.split('_')
        if 'lem' in parts:
            info['lemmatized'] = True
        if 'postag' in parts:
            info['pos_tagged'] = True
        if 'phrase' in parts:
            info['phrase_detection'] = True
        
        # Extract dimensions if present
        for part in parts:
            if part.isdigit():
                info['dimensions'] = int(part)
                break
        
        # Extract training algorithm
        if 'cbow' in parts:
            info['algorithm'] = 'cbow'
        elif 'skip' in parts:
            info['algorithm'] = 'skip-gram'
        
        # Extract cutoff if present
        for part in parts:
            if part.startswith('cut'):
                try:
                    info['cutoff'] = int(part[3:])
                except:
                    pass
        
        # Save the info
        with open(os.path.join(model_dir, "info.json"), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Model successfully saved to {model_dir}")
        print(f"Vocabulary size: {vocab_size}")
        if total_tokens:
            print(f"Total tokens: {total_tokens:,}")
        print(f"Vector size: {vector_size}")
        
        return model_dir
    
    finally:
        # Clean up the temporary file regardless of success or failure
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print("Removed temporary download file to save space")

