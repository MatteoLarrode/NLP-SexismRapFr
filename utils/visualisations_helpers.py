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