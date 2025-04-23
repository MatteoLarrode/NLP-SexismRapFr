# ========================================================================
# === Helper functions for the choice of words (target and attributes) ===
# ========================================================================
import pandas as pd

# Define the attribute and target word lists for French rap
attribute_words = {
    'M': ['homme', 'mec', 'gars', 'frère', 'il', 'père', 'papa', 'oncle', 'bro', 'cousin'],
    'F': ['femme', 'meuf', 'fille', 'sœur', 'elle', 'mère', 'maman', 'tante', 'miss', 'cousine']
}

target_words = {
    'B1_career_family': {
        'X': ['business', 'patron', 'patronne', 'argent', 'money', 'travail', 'boss', 'cash', 'bureau', 'carrière'],
        'Y': ['foyer', 'parents', 'maison', 'enfant', 'enfants', 'famille', 'mariage', 'marier']
    },
    'B2_mathsci_arts': {
        'X': ['calcul', 'logique', 'science', 'chiffres', 'physique', 'maths', 'chimie'],
        'Y': ['poésie', 'art', 'danse', 'littérature', 'chanson', 'peinture']
    },
    'B3_intel_appearance': {
        'X': ['brillant', 'brillante', 'intelligent', 'intelligente', 'stratège', 'cerveau', 'sage', 'lucide', 'génie'],
        'Y': ['beau', 'belle', 'mince', 'moche', 'laid', 'laide', 'joli', 'jolie', 'maigre', 'gros', 'grosse', 'corps']
    },
    'B4_strength_weakness': {
        'X': ['puissant', 'puissante', 'puissance',  'fort', 'forte', 'force'],
        'Y': ['faible', 'fragile' , 'doux' ,'douce', 'sensible', 'peur', 'vulnérable']
    }
}

# Gendered word pairs
gendered_pairs = [
    ('patron', 'patronne'),
    ('intelligent', 'intelligente'),
    ('beau', 'belle'),
    ('laid', 'laide'),
    ('joli', 'jolie'),
    ('gros', 'grosse'),
    ('fort', 'forte'),
    ('puissant', 'puissante'),
    ('doux', 'douce'),
]

def compare_gendered_word_similarities(models, gendered_pairs):
    """
    Calculate cosine similarities between gendered word pairs across different models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of embedding models {model_name: model}
    gendered_pairs : list of tuples
        List of (masculine_form, feminine_form) word pairs
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with similarity scores across all models
    """
    # Prepare results
    results = []
    
    for model_name, model in models.items():
        # Process embeddings if needed
        embeddings = model
        if hasattr(model, 'wv'):
            embeddings = model.wv
            
        for masc, fem in gendered_pairs:
            # Check if both words exist in the model
            if masc in embeddings and fem in embeddings:
                # Calculate cosine similarity
                similarity = embeddings.similarity(masc, fem)
                
                # Store result
                results.append({
                    'model': model_name,
                    'masculine': masc,
                    'feminine': fem,
                    'pair': f"{masc}/{fem}",
                    'similarity': similarity
                })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add summary statistics per model
    if not df.empty:
        model_stats = df.groupby('model')['similarity'].agg(['mean', 'min', 'max', 'count']).reset_index()
        print("Model Statistics:")
        print(model_stats)
    
    return df

def check_rap_frequencies(corpus, all_words):
    """
    Check frequency of words in the rap corpus.
    
    Parameters:
    -----------
    corpus : list or DataFrame
        The rap corpus (adapts to format)
    all_words : list
        List of words to check
        
    Returns:
    --------
    DataFrame
        Words and their frequencies
    """
    # Initialize results
    results = []
    
    # Handle different corpus formats
    if isinstance(corpus, pd.DataFrame):
        # DataFrame format
        for word in all_words:
            # Check if word exists in corpus
            exists = word in corpus.columns
            
            # Get frequency if exists
            if exists:
                frequency = corpus[word].sum()
            else:
                frequency = 0
                
            results.append({
                'word': word,
                'in_corpus': exists,
                'frequency': frequency
            })
    elif isinstance(corpus, list):
        # List format (assuming list of documents/songs)
        from collections import Counter
        
        # Count all words in the corpus
        word_counts = Counter()
        for document in corpus:
            # Handle different document formats (str, list, dict)
            if isinstance(document, str):
                # Split string into words
                words = document.lower().split()
                word_counts.update(words)
            elif isinstance(document, list):
                # Already a list of words
                word_counts.update([w.lower() if isinstance(w, str) else w for w in document])
            elif isinstance(document, dict):
                # Dictionary with word counts
                word_counts.update(document)
        
        # Check each target word
        for word in all_words:
            # Case-insensitive check
            word_lower = word.lower()
            exists = word_lower in word_counts
            frequency = word_counts.get(word_lower, 0)
            
            results.append({
                'word': word,
                'in_corpus': exists,
                'frequency': frequency
            })
    else:
        raise TypeError("Corpus must be a DataFrame or a list")
    
    # Convert to DataFrame and sort by frequency
    df = pd.DataFrame(results)
    df = df.sort_values('frequency', ascending=False)
    
    return df

def check_model_existence(models, all_words):
    """
    Check if words exist in each model.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models {name: model}
    all_words : list
        List of words to check
        
    Returns:
    --------
    DataFrame
        Words and their existence in each model
    """
    import pandas as pd
    
    # Initialize results
    results = []
    
    for word in all_words:
        result = {'word': word}
        
        # Check each model
        for model_name, model in models.items():
            # Get embeddings
            if hasattr(model, 'wv'):
                embeddings = model.wv
            else:
                embeddings = model
                
            # Check if word exists
            if hasattr(embeddings, 'key_to_index'):
                result[model_name] = word in embeddings.key_to_index
            elif isinstance(embeddings, dict):
                result[model_name] = word in embeddings
            else:
                result[model_name] = False
                
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

# Collect all words
def get_all_words(attribute_words, target_words):
    """Get all unique words from attribute and target word sets"""
    all_words = []
    
    # Add attribute words
    for words in attribute_words.values():
        all_words.extend(words)
        
    # Add target words
    for category in target_words.values():
        for words in category.values():
            all_words.extend(words)
            
    # Return unique words
    return list(set(all_words))