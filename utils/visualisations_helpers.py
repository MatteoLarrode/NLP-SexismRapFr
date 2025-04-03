# ===============================================
# === Helper functions for data visualisation ===
# ===============================================
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
import pandas as pd

def set_visualization_style():
    plt.style.use('seaborn-v0_8-colorblind')
    #font_path = '/Users/matteolarrode/Library/Fonts/cmunss.ttf'
    #font_manager.fontManager.addfont(font_path)
    #prop = font_manager.FontProperties(fname=font_path)
    #plt.rcParams['font.family'] = 'sans-serif'
    #plt.rcParams['font.sans-serif'] = prop.get_name()
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

# Function to visualize similarity changes over time for a word pair
def plot_similarity_over_time(models, word1, word2):
    """Plot how the similarity between two words changes across different decade models."""
    decades = []
    similarities = []
    
    # Only use decade models, not the 'all' model
    decade_models = {k: v for k, v in models.items() if k != 'all'}
    
    for model_name, model in sorted(decade_models.items()):
        if word1 in model.wv and word2 in model.wv:
            similarity = model.wv.similarity(word1, word2)
            decades.append(model_name)
            similarities.append(similarity)
    
    if not decades:
        print(f"No valid results for '{word1}' and '{word2}' across decade models")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(decades, similarities, marker='o', linestyle='-', linewidth=2)
    plt.title(f"Similarity between '{word1}' and '{word2}' over time")
    plt.xlabel('Decade')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
