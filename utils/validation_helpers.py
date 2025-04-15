# ==========================================================
# === Helper functions for validation of word embeddings ===
# ==========================================================
import requests
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from adjustText import adjust_text
from nltk.corpus import stopwords

from utils.visualisations_helpers import set_visualization_style

# ===== Utility functions =====
def load_model(model_dir):
    """
    Load a word2vec model from the specified directory using the new storage system.
    
    Args:
        model_dir (str): Path to the model directory
        
    Returns:
        model: The loaded Word2Vec model
    """
    from gensim.models import Word2Vec
    
    # Load the model
    model_path = os.path.join(model_dir, "model")
    model = Word2Vec.load(model_path)
    
    # Load model info if available
    info_path = os.path.join(model_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            model.info = json.load(f)
            
    # Set model name attribute
    model.name = os.path.basename(model_dir)
    
    return model

def load_models_from_directory(base_dir='models'):
    """
    Load all models from the specified directory.
    
    Args:
        base_dir (str): Base directory containing model directories
        
    Returns:
        dict: Dictionary with model names as keys and loaded models as values
    """
    models = {}
    
    # Get all subdirectories in the base directory
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for model_dir in subdirs:
        full_path = os.path.join(base_dir, model_dir)
        
        # Check if this directory contains a model file
        if os.path.exists(os.path.join(full_path, "model")):
            try:
                # Load the model
                model = load_model(full_path)
                models[model_dir] = model
                print(f"Loaded model: {model_dir}")
            except Exception as e:
                print(f"Error loading model from {model_dir}: {e}")
    
    return models

# ===== Clustering and Visualization =====
def kmeans_cluster_embeddings(model, n_words=200, n_clusters=5, random_state=35, 
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

# ===== Word Similarity Validation =====
def create_similarity_dict(csv_path):
    """
    Creates a dictionary from the similarity CSV data where:
    - Keys are tuples of French word pairs (mot1, mot2)
    - Values are the human mean similarity ratings
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary with word pairs as keys and similarity ratings as values
    """
    # Read the CSV file
    df = pd.read_csv(csv_path, header=0)
    
    # Rename columns for clarity
    df.columns = ['word1_en', 'word2_en', 'human_mean', 'mot1_fr', 'mot2_fr']

    # Remove rows where mot1_fr and mot2_fr are the same
    df = df[df['mot1_fr'] != df['mot2_fr']]
    
    # Create dictionary with French word pairs as keys and human mean as values
    similarity_dict = {}
    for _, row in df.iterrows():
        # Create tuple of French words as key
        key = (row['mot1_fr'], row['mot2_fr'])
        # Use human mean as value
        value = row['human_mean']
        similarity_dict[key] = value
    
    return similarity_dict
    
def add_model_similarities(similarity_dict, model):
    """
    Add model-computed similarities to the dictionary.
    
    Args:
        similarity_dict: Dictionary with (word1, word2) tuples as keys and expected similarity as values
        model: Word2Vec model to evaluate
        
    Returns:
        dict: Original dictionary with an additional tuple value containing 
              (expected_similarity, model_similarity, in_vocab)
    """
    result_dict = {}
    missing_pairs = []
    
    for (word1, word2), expected_similarity in similarity_dict.items():
        # Check if both words are in the vocabulary
        in_vocab = word1 in model.wv and word2 in model.wv
        
        if in_vocab:
            # Calculate cosine similarity
            model_similarity = model.wv.similarity(word1, word2)
            result_dict[(word1, word2)] = (expected_similarity, model_similarity, True)
        else:
            # If one or both words are not in vocab, mark as not in vocabulary
            result_dict[(word1, word2)] = (expected_similarity, None, False)
            missing_words = []
            if word1 not in model.wv:
                missing_words.append(word1)
            if word2 not in model.wv:
                missing_words.append(word2)
            missing_pairs.append(f"({word1}, {word2}) - missing: {', '.join(missing_words)}")
    
    # Print statistics about missing words
    if missing_pairs:
        print(f"Number of word pairs not in vocabulary: {len(missing_pairs)} / {len(similarity_dict)} ({len(missing_pairs)/len(similarity_dict)*100:.2f}%)")
    
    return result_dict, missing_pairs

def validate_embeddings_on_similarities(result_dict):
    """
    Validate word embeddings by calculating correlation between expected and model similarities.
    
    Args:
        result_dict: Dictionary with (word1, word2) tuples as keys and 
                    (expected_similarity, model_similarity, in_vocab) as values
        
    Returns:
        tuple: (pearson_corr, pearson_p, spearman_corr, spearman_p, valid_pairs, total_pairs)
    """
    # Extract pairs where both words are in the vocabulary
    valid_pairs = [(expected, model) for (expected, model, in_vocab) in result_dict.values() 
                  if in_vocab and not np.isnan(expected) and not np.isnan(model)]
    
    if len(valid_pairs) < 2:
        print("Not enough valid pairs to calculate correlation.")
        return None, None, None, None, 0, len(result_dict)
    
    # Split into expected and model similarities
    expected_similarities = [pair[0] for pair in valid_pairs]
    model_similarities = [pair[1] for pair in valid_pairs]
    
    # Calculate correlation metrics
    pearson_corr, pearson_p = pearsonr(expected_similarities, model_similarities)
    spearman_corr, spearman_p = spearmanr(expected_similarities, model_similarities)
    
    print(f"Evaluation results:")
    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    
    return pearson_corr, pearson_p, spearman_corr, spearman_p, len(valid_pairs), len(result_dict)

def plot_similarity_correlation(result_dict, save_path=None, show_plot=True):
    """
    Plot the correlation between expected (ground truth) and model-computed similarities.
    
    Args:
        result_dict: Dictionary with (word1, word2) tuples as keys and 
                    (expected_similarity, model_similarity, in_vocab) as values
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot (default: True)
    
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

    # Use constom style
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
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show the plot if requested
    if show_plot:
        plt.show()

    else :
        plt.close()
    
    return

def validate_models_similarity_task(models, show_plot=True):
    """
    Evaluate all models on word similarity task.
    
    Args:
        models (dict): Dictionary of models to evaluate
        results_dir (str): Directory to save results
        
    Returns:
        pd.DataFrame: DataFrame with similarity results for all models
    """    
    # Load similarity dataset
    similarity_dict = create_similarity_dict("data/wordsim353-fr.csv")
    
    # Results list for all models
    results_list = []
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} on word similarity...")
        
        # Add model similarities
        result_dict, missing_pairs = add_model_similarities(similarity_dict, model)
        
        # Validate embeddings
        pearson_corr, pearson_p, spearman_corr, spearman_p, valid_pairs, total_pairs = validate_embeddings_on_similarities(result_dict)
        
        # Plot correlation if valid
        if pearson_corr is not None:
            plot_similarity_correlation(result_dict, save_path=f"figs/similarity_corr/{model_name}_similarity_correlation.png", show_plot=show_plot)
        
        # Add to results list
        results_list.append({
            'Model': model_name,
            'Valid_Pairs': valid_pairs,
            'Total_Pairs': total_pairs,
            'Coverage': valid_pairs / total_pairs if total_pairs > 0 else 0,
            'Pearson_Correlation': pearson_corr,
            'Pearson_P_Value': pearson_p,
            'Spearman_Correlation': spearman_corr,
            'Spearman_P_Value': spearman_p
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results_list)
    
    return results_df

# ===== Word Analogy Validation =====
def download_analogy_dataset(url):
    """Download the analogy dataset."""
    response = requests.get(url)
    if response.status_code == 200:
        response.encoding = 'utf-8'  # Ensure proper handling of accents and special characters
        return response.text
    else:
        print(f"Failed to download dataset: {response.status_code}")
        return None

def parse_analogy_dataset(data_text):
    """Parse the analogy dataset into categories and questions."""
    categories = defaultdict(list)
    current_category = None
    
    for line in data_text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        if line.startswith(':'):
            current_category = line[2:]  # Remove ': ' prefix
        if current_category:
                words = line.split()
                if len(words) == 4:  # Ensure valid analogy format
                    a, b, c, expected = words
                    
                    # Filter out cases where expected word is the same as a, b, or c
                    if expected.lower() not in [a.lower(), b.lower(), c.lower()]:
                        categories[current_category].append(words)
    
    return categories

def perform_analogy_test(model, a, b, c):
    """
    Perform the analogy test: a is to b as c is to ?
    Using the 3CosAdd method: b - a + c
    
    Returns the predicted word and its similarity score.
    """
    try:
        # Get the word vectors for the input words
        vec_a = model.wv.get_vector(a, norm=True)
        vec_b = model.wv.get_vector(b, norm=True)
        vec_c = model.wv.get_vector(c, norm=True)
        
        # Find the most similar words to this vector
        result = model.wv.most_similar(positive=[vec_b, vec_c], negative=[vec_a], topn=10)
        
        # Return the top prediction
        if result:
            # Return the top 3 predictions
            return result[:3] if result else None
        else:
            return None
        
    except KeyError:
        # Handle case where one of the words is not in the vocabulary
        return None
    
def evaluate_model_on_analogies(model, categories, verbose=True, save_correct=False):
    """
    Evaluate a model on all analogy categories.
    Returns accuracy for each category and overall accuracy.
    
    Parameters:
    - model: The Word2Vec model to evaluate
    - categories: Dictionary of categories and their questions
    - verbose: Whether to print progress
    - save_correct: Whether to save correct analogies
    """
    results = {}
    total_correct = 0
    total_questions = 0
    skipped_questions = 0
    correct_analogies = []
    
    for category, questions in categories.items():
        correct = 0
        answered = 0
        category_correct = []
        
        if verbose:
            print(f"Evaluating category: {category}")
            iterator = tqdm(questions)
        else:
            iterator = questions
            
        for a, b, c, expected in iterator:
            # Skip if any word is not in the vocabulary
            if not all(word in model.wv for word in [a, b, c, expected]):
                skipped_questions += 1
                continue
                
            prediction = perform_analogy_test(model, a, b, c)
            
            if prediction and any(pred[0].lower() == expected.lower() for pred in prediction):
                correct += 1
                
                # Save correct analogy if requested
                if save_correct:
                    category_correct.append({
                        'analogy': f"{a} : {b} :: {c} : {expected}",
                        'similarity_score': prediction[0][1]
                    })
            
            answered += 1
        
        if answered > 0:
            accuracy = correct / answered
            results[category] = {
                'accuracy': accuracy,
                'answered': answered,
                'correct': correct
            }
            
            # Save correct analogies for this category
            if save_correct and category_correct:
                correct_analogies.append({
                    'category': category,
                    'examples': category_correct
                })
            
            total_correct += correct
            total_questions += answered
            
            if verbose:
                print(f"  Accuracy: {accuracy:.4f} ({correct}/{answered})")
        else:
            results[category] = {
                'accuracy': None, # No questions answered
                'answered': 0,
                'correct': 0
            }
            
            if verbose:
                print("  No questions could be answered (words not in vocabulary)")
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    
    if verbose:
        print(f"\nOverall accuracy: {overall_accuracy:.4f} ({total_correct}/{total_questions})")
        print(f"Skipped questions: {skipped_questions}")
    
    return results, overall_accuracy, skipped_questions, correct_analogies

def save_correct_analogies(correct_analogies, save_file):
    """
    Save correct analogies to a file.
    
    Parameters:
    - correct_analogies: List of dictionaries with correct analogies
    - save_file: Path to save the file
    """
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write("# Correct Analogies\n\n")
        
        for category_data in correct_analogies:
            category = category_data['category']
            examples = category_data['examples']
            
            f.write(f"## Category: {category}\n")
            f.write(f"Total correct: {len(examples)}\n\n")
            
            for i, example in enumerate(examples):
                f.write(f"{i+1}. {example['analogy']} (similarity: {example['similarity_score']:.4f})\n")
            
            f.write("\n" + "-" * 50 + "\n\n")
    
    print(f"Saved correct analogies to {save_file}")

def validate_models_analogy_task(models, save_correct=False, results_dir="results"):
    """
    WRAPPER: Evaluate word embeddings using the analogy test.
    Works with a single model or a dictionary of models.
    
    Parameters:
    - models: A single Word2Vec model or a dictionary of models (key: model_name, value: model)
    - dataset_url: URL of the analogy dataset
    - save_correct: Whether to save correct analogies
    - results_dir: Directory to save results
    
    Returns:
    - For a single model: Dictionary with evaluation results
    - For multiple models: DataFrame with comparison results
    """        
    # Download the dataset
    dataset_url="https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-fr.txt"

    print("Downloading analogy dataset...")
    data_text = download_analogy_dataset(dataset_url)
    
    # Parse the dataset
    categories = parse_analogy_dataset(data_text)
    print(f"Parsed {len(categories)} categories with a total of {sum(len(q) for q in categories.values())} questions")
    
    if not isinstance(models, dict):
        model = models
        model_name = getattr(model, 'name', 'model')  # Use model.name if available, otherwise 'model'
        models = {model_name: model}
    
    # List to store results for all models
    results_list = []
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        
        # Evaluate model (no verbose)
        results, overall_accuracy, skipped, correct_analogies = evaluate_model_on_analogies(
            model, categories, verbose=False, save_correct=save_correct)
        
        # Save correct analogies if requested
        if save_correct and correct_analogies:
            save_file = f"{results_dir}/correct_analogies/correct_analogies_{model_name}.md"
            save_correct_analogies(correct_analogies, save_file)
        
        # Create a dictionary with model results
        model_results = {
            'Model': model_name,
            'Overall_Accuracy': overall_accuracy,
            'Skipped_Questions': skipped
        }
        
        # Add category-specific results
        for category, data in results.items():
            model_results[f"{category}"] = data['accuracy']
        
        # Add to results list
        results_list.append(model_results)
        
        # Print brief summary
        print(f"  Accuracy: {overall_accuracy:.4f}, Skipped: {skipped}")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results_list)
    
    return results_df

def print_sorted_categories(analogy_results_df, model_name):
    """
    Print categories sorted by accuracy for a specific model's analogy results.
    
    Args:
        analogy_results_df (pd.DataFrame): DataFrame with analogy results
        model_name (str): Name of the model to analyze
    """
    if len(analogy_results_df) == 0:
        print("Empty results DataFrame")
        return
        
    # Check if the model exists in the results
    if model_name not in analogy_results_df['Model'].values:
        print(f"Model '{model_name}' not found in results")
        return
        
    # Get the row for the specified model
    model_row = analogy_results_df[analogy_results_df['Model'] == model_name].iloc[0]
    
    # Get all category columns
    category_columns = [col for col in analogy_results_df.columns 
                       if col not in ['Model', 'Overall_Accuracy', 'Skipped_Questions']]
    
    if not category_columns:
        print("No category columns found in results")
        return
    
    # Get category accuracies from the model row, filtering out None values
    categories_acc = [(cat, model_row[cat]) for cat in category_columns 
                     if model_row[cat] is not None]
    
    # Sort by accuracy (descending)
    sorted_categories = sorted(categories_acc, key=lambda x: x[1], reverse=True)
    
    # Print all categories in order
    print(f"\nCategory Performance for model '{model_name}' (best to worst):")
    for category, acc in sorted_categories:
        print(f"  {category}: {acc:.4f}")