# ===============================================
# === Helper functions for data visualisation ===
# ===============================================
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from adjustText import adjust_text
import numpy as np
from nltk.corpus import stopwords


def set_visualization_style():
    plt.style.use('seaborn-v0_8-colorblind')
    font_path = '/Users/matteolarrode/Library/Fonts/cmunss.ttf'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 1.5,
        'lines.markersize': 8,
        'figure.figsize': (10, 6),
        'axes.grid': False, 
        'axes.spines.top': False,  # Remove top spine
        'axes.spines.right': False,  # Remove right spine
    })

def plot_word_comparisons(models, base_word, comparison_words):
    """Plot similarities between a base word and multiple comparison words across models."""
    results = {}
    valid_models = []
    
    for model_name, model in models.items():
        if base_word in model.wv:
            model_results = {}
            for comp_word in comparison_words:
                if comp_word in model.wv:
                    similarity = model.wv.similarity(base_word, comp_word)
                    model_results[comp_word] = similarity
            
            if model_results:  # Only add if we got results
                results[model_name] = model_results
                valid_models.append(model_name)
    
    if not results:
        print(f"No valid results for '{base_word}' with the comparison words")
        return
    
    # Create a DataFrame for plotting
    df_list = []
    for model_name in valid_models:
        for word, similarity in results[model_name].items():
            df_list.append({
                'model': model_name,
                'comparison_word': word,
                'similarity': similarity
            })
    
    df = pd.DataFrame(df_list)
    
    # Plot
    plt.figure(figsize=(14, 8))
    sns.barplot(x='comparison_word', y='similarity', hue='model', data=df)
    plt.title(f"Similarity to '{base_word}' across models")
    plt.xlabel('Comparison Word')
    plt.ylabel('Cosine Similarity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Model')
    plt.show()

def visualise_embeddings(model, n_words=500, random_state=42, perplexity=30, 
                          categories=None, 
                          word_list=None, save_path=None):
    """
    Visualize word embeddings using t-SNE
    
    Parameters:
    -----------
    model : Word2Vec model
        The trained word2vec model
    n_words : int
        Number of most frequent words to visualize
    random_state : int
        Random seed for reproducibility
    perplexity : int
        t-SNE perplexity parameter
    categories : dict
        Dictionary mapping category names to lists of words
        Example: {'artists': ['mc_solaar', 'iam', 'ntz'], 'slang': ['flow', 'kickeur']}
    word_list : list
        Specific list of words to visualize (overrides n_words)
    save_path : str
        Path to save the visualization
    """
    french_stopwords = set(stopwords.words('french'))
    filter_words = list(french_stopwords)
    
    # Get words to visualize
    if word_list is not None:
        # Use provided word list
        words = [w for w in word_list if w in model.wv.key_to_index]
    else:
        # Get most common words - using the vocabulary instead of most_common
        # Sort by frequency if available, otherwise just use the keys
        if hasattr(model.wv, 'get_vecattr'):
            # For newer gensim versions
            word_freq = {word: model.wv.get_vecattr(word, 'count') 
                        for word in model.wv.key_to_index}
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            words = [word for word, _ in sorted_words 
                    if word not in filter_words][:n_words]
        else:
            # Fallback if count information is not available
            words = list(model.wv.key_to_index.keys())
            if len(words) > n_words:
                words = words[:n_words * 2]
            words = [word for word in words if word not in filter_words][:n_words]
    
    # Get word vectors
    word_vectors = np.array([model.wv[word] for word in words])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    coordinates = tsne.fit_transform(word_vectors)

    # Apply custom style
    set_visualization_style()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'word': words,
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'category': ['other'] * len(words)
    })
    
    # Update categories if provided
    if categories is not None:
        for category, category_words in categories.items():
            mask = df['word'].isin(category_words)
            df.loc[mask, 'category'] = category
    
    # Get unique categories and ensure 'other' is at the end
    unique_categories = sorted(df['category'].unique(), key=lambda x: x == 'other')
    colors = cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    
    # Plot by category
    texts = []
    for i, category in enumerate(unique_categories):
        mask = df['category'] == category
        subset = df[mask]

        # Format category for legend: capitalize and replace underscores with spaces
        formatted_category = category.replace('_', ' ').capitalize()

        # Set color: grey for general category, rainbow colors for others
        if category == 'other':
            point_color = 'grey'
        else:
            point_color = colors[i]
        
        # Plot points
        plt.scatter(subset['x'], subset['y'], 
                    c=[point_color], 
                    label=formatted_category, 
                    alpha=0.7)
        
        # Add text labels
        for _, row in subset.iterrows():
            texts.append(plt.text(row['x'], row['y'], row['word'], 
                                  fontsize=11, alpha=0.8))
    
    # Adjust text to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    # Add legend and labels with improved formatting
    legend = plt.legend(loc='upper right', frameon=True, framealpha=0.9, fancybox=True)
    legend.get_frame().set_edgecolor('lightgray')
    
    # Add legend and labels
    plt.title("t-SNE visualization of word embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Remove ticks
    plt.xticks([])
    plt.yticks([])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()