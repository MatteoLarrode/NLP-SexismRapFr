# ========================================================================
# === Helper functions for the choice of words (target and attributes) ===
# ========================================================================

# Define the attribute and target word lists for French rap
attribute_words = {
    'M': ['homme', 'mec', 'gars', 'frère', 'il', 'père', 'papa', 'oncle', 'grand-père', 'bro', 'cousin'],
    'F': ['femme', 'meuf', 'fille', 'sœur', 'elle', 'mère', 'maman', 'tante', 'grand-mère', 'miss', 'cousine']
}

target_words = {
    'B1_career_family': {
        'X': ['business', 'patron', 'patronne', 'argent', 'money', 'travail', 'boss', 'cash', 'hustle', 'bureau', 'carrière'],
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
        'X': ['confiant','confiante', 'confiance' , 'puissant', 'puissante', 'puissance',  'fort', 'forte', 'force', 'dominant', 'dominante', 'dominance'],
        'Y': ['faible', 'fragile', 'timide' , 'doux' ,'douce', 'sensible', 'soumis', 'soumise',  'peur', 'vulnérable']
    }
}

# Gendered word pairs
gendered_pairs = [
    ('patron', 'patronne'),
    ('brilliant', 'brillante'),
    ('intelligent', 'intelligente'),
    ('beau', 'belle'),
    ('laid', 'laide'),
    ('joli', 'jolie'),
    ('gros', 'grosse'),
    ('fort', 'forte'),
    ('puissant', 'puissante'),
    ('dominant', 'dominante'),
    ('confiant', 'confiante'),
    ('doux', 'douce'),
    ('soumis', 'soumise')
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

