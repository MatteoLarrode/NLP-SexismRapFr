# ==========================================================
# === Helper functions for validation of word embeddings ===
# ==========================================================
import requests
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os
import pandas as pd
from collections import defaultdict

# ===== Download and parse datasets =====
def download_analogy_dataset(url):
    """Download the analogy dataset."""
    response = requests.get(url)
    if response.status_code == 200:
        response.encoding = 'utf-8'  # Ensure proper handling of accents and special characters
        return response.text
    else:
        print(f"Failed to download dataset: {response.status_code}")
        return None
    
def download_similarity_dataset(url):
    """Download the similarity dataset."""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download dataset: {response.status_code}")
    
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel(url)
    return df

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

def parse_similarity_dataset(df):
    """
    Parse the similarity dataset into a dictionary format.
    
    Args:
        df: Pandas DataFrame containing the similarity dataset
        
    Returns:
        dict: Dictionary with (word1, word2) tuples as keys and expected similarity as values
    """
    similarity_dict = {}
    
    for _, row in df.iterrows():
        word_pair = row['wordPairs']
        similarity = row['MeanPairSimilarity']
        
        # Split the word pair by dash
        words = word_pair.split('-')
        
        # Only process if we have exactly two words
        if len(words) == 2:
            word1, word2 = words[0].strip(), words[1].strip()
            similarity_dict[(word1, word2)] = similarity
        else:
            print(f"Skipping word pair '{word_pair}' - could not parse properly")
    
    return similarity_dict
    
# ===== Word similarity validation =====
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
        print(f"Number of word pairs not in vocabulary: {len(missing_pairs)} / {len(similarity_dict)}")
        print(f"Examples of missing pairs:")
        for pair in missing_pairs[:5]:
            print(f"  {pair}")
        if len(missing_pairs) > 5:
            print(f"  ... and {len(missing_pairs) - 5} more")
    
    return result_dict

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
    print(f"Valid pairs: {len(valid_pairs)} / {len(result_dict)} ({len(valid_pairs)/len(result_dict)*100:.2f}%)")
    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    
    return pearson_corr, pearson_p, spearman_corr, spearman_p, len(valid_pairs), len(result_dict)

# ===== Analogy validation =====
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

def validate_word_embeddings_on_analogies(models, dataset_url="https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-fr.txt", save_correct=False, results_dir="results"):
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
        
        # Evaluate model (verbose only if we have a single model)
        verbose = (len(models) == 1)
        results, overall_accuracy, skipped, correct_analogies = evaluate_model_on_analogies(
            model, categories, verbose=verbose, save_correct=save_correct)
        
        # Save correct analogies if requested
        if save_correct and correct_analogies:
            save_file = f"{results_dir}/correct_analogies_{model_name}.md"
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