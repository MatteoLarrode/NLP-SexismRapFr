# ===============================================
# === Helper functions for data visualisation ===
# ===============================================
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
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

def kmeans_cluster_embeddings(model, n_words=200, n_clusters=5, random_state=42, 
                              word_list=None, save_path=None):
    """
    Perform k-means clustering on word embeddings and visualize the results
    
    Parameters:
    -----------
    model : Word2Vec model
        The trained word2vec model
    n_words : int
        Number of most frequent words to include if word_list is not provided
    n_clusters : int
        Number of clusters for k-means
    random_state : int
        Random seed for reproducibility
    word_list : list
        Specific list of words to analyze (overrides n_words)
    save_path : str
        Path to save the visualization
        
    Returns:
    --------
    dict : Cluster information including words in each cluster
    """
    french_stopwords = set(stopwords.words('french'))
    filter_words = list(french_stopwords)
    
    # Get words to analyze
    if word_list is not None:
        # Use provided word list
        words = [w for w in word_list if w in model.wv.key_to_index and w not in filter_words]
    else:
        # Get words from the model vocabulary
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
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(word_vectors)
    
    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(words)//4), 
                max_iter=2000, init='pca')
    coordinates = tsne.fit_transform(word_vectors)
    
    # Create cluster information
    cluster_info = {i: [] for i in range(n_clusters)}
    for word, cluster_id in zip(words, clusters):
        cluster_info[cluster_id].append(word)
    
    # Create dataframe for visualization
    df = pd.DataFrame({
        'word': words,
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'cluster': clusters
    })
    
    # Visualize
    set_visualization_style()
    plt.figure(figsize=(12, 10))
    
    # Get cluster colors
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    texts = []
    for i in range(n_clusters):
        cluster_df = df[df['cluster'] == i]
        plt.scatter(cluster_df['x'], cluster_df['y'], 
                    c=[colors[i]], 
                    label=f"Cluster {i+1}", 
                    alpha=0.7)
        
        # Add text labels
        for _, row in cluster_df.iterrows():
            texts.append(plt.text(row['x'], row['y'], row['word'], 
                                 fontsize=9, alpha=0.8))
    
    # Adjust text to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    # Add legend with better formatting
    legend = plt.legend(loc='upper right', frameon=True, framealpha=0.9, 
                        fancybox=True, title="Clusters")
    legend.get_frame().set_edgecolor('lightgray')
    
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)

    # Remove ticks
    plt.xticks([])
    plt.yticks([])
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return cluster_info

def plot_similarity_correlation(result_dict, save_path=None):
    """
    Plot the correlation between expected (ground truth) and model-computed similarities.
    
    Args:
        result_dict: Dictionary with (word1, word2) tuples as keys and 
                    (expected_similarity, model_similarity, in_vocab) as values
        save_path: Path to save the figure (optional)
    
    Returns:
        matplotlib figure object
    """
    # Extract pairs where both words are in the vocabulary
    valid_pairs = [(expected, model, words) for words, (expected, model, in_vocab) in result_dict.items() 
                  if in_vocab and not np.isnan(expected) and not np.isnan(model)]
    
    if len(valid_pairs) < 2:
        print("Not enough valid pairs to create a correlation plot.")
        return None
    
    # Split into expected and model similarities
    expected_similarities = [pair[0] for pair in valid_pairs]
    model_similarities = [pair[1] for pair in valid_pairs]
    word_pairs = [pair[2] for pair in valid_pairs]
    
    # Calculate Pearson correlation for the plot
    pearson_corr, p_value = pearsonr(expected_similarities, model_similarities)

    # Use constim style
    set_visualization_style()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Create scatter plot
    plt.scatter(expected_similarities, model_similarities, alpha=0.6)
    
    # Add regression line
    slope, intercept = np.polyfit(expected_similarities, model_similarities, 1)
    x_line = np.linspace(min(expected_similarities), max(expected_similarities), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='red', linestyle='--')
    
    # Add correlation coefficient to the plot
    plt.text(0.05, 0.95, f'Pearson r = {pearson_corr:.4f} (p = {p_value:.4f})', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.xlabel('Translated WordSim 353 Similarity')
    plt.ylabel('Cosine Similarity')
    
    # Improve plot aesthetics
    plt.tight_layout()

    plt.show()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return