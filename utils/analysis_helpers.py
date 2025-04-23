# =========================================================
# === Helper functions for the word embeddings analysis ===
# =========================================================
import numpy as np
import itertools
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple

from utils.word_choice_helpers import attribute_words, target_words

# ===== WEAT experiment =====
class GenderBiasWEATAnalyser:
    def __init__(self, embeddings, embedding_dimension = None):
        """
        Initialize the analyser with word embeddings
        
        Parameters:
        -----------
        embeddings : dict or KeyedVectors or Word2Vec
            Word embeddings in one of these formats:
            - dict: A dictionary mapping words to numpy vectors
            - KeyedVectors: A gensim KeyedVectors object
            - Word2Vec: A gensim Word2Vec model
        embedding_dimension : int, optional
            The dimension of the word vectors (only needed if embeddings is a dict)
        """
        self.embeddings = self._process_embeddings(embeddings, embedding_dimension)
        self.results = {}
        self.effect_sizes = {}
        self.p_values = {}

    def _process_embeddings(self, embeddings, embedding_dimension):
        """
        Process different input embedding formats into a standard format
        
        Parameters:
        -----------
        embeddings : dict or KeyedVectors or Word2Vec
            Word embeddings in various formats
        embedding_dimension : int, optional
            The dimension of word vectors if embeddings is a dict
            
        Returns:
        --------
        dict
            Dictionary mapping words to numpy vectors
        """
        # If embeddings is a Word2Vec model, extract its KeyedVectors
        if hasattr(embeddings, 'wv'):
            embeddings = embeddings.wv

        # If embeddings is a KeyedVectors object, convert to dict
        if hasattr(embeddings, 'key_to_index') and hasattr(embeddings, 'get_vector'):
            embedding_dict = {}
            for word in embeddings.key_to_index:
                embedding_dict[word] = embeddings.get_vector(word)
            return embedding_dict
        
        # If embeddings is already a dict mapping words to vectors, use it directly
        if isinstance(embeddings, dict):
            # Verify that values are numpy arrays with proper dimensions
            if embedding_dimension is not None:
                for word, vector in embeddings.items():
                    if len(vector) != embedding_dimension:
                        raise ValueError(f"Vector for word '{word}' has dimension {len(vector)}, "
                                        f"expected {embedding_dimension}")
            return embeddings
            
        raise TypeError("Embeddings must be a dict mapping words to vectors, "
                      "a gensim KeyedVectors object, or a Word2Vec model")
    
    def cosine_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate the cosine similarity between two words
        
        Parameters:
        -----------
        word1, word2 : str
            Words to compare
            
        Returns:
        --------
        float
            Cosine similarity between the two words
        """
        if word1 not in self.embeddings or word2 not in self.embeddings:
            return np.nan
            
        vec1 = self.embeddings[word1]
        vec2 = self.embeddings[word2]
        
        # Calculate cosine similarity manually
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return np.nan
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def association_measure(self, target_word: str, A: List[str], B: List[str]) -> float:
        """
        Calculate the association measure s(w, A, B) as defined in WEAT methodology
        s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b)
        
        Parameters:
        -----------
        word : str
            Target word
        A : List[str]
            First list of attribute words (e.g., male terms)
        B : List[str]
            Second list of attribute words (e.g., female terms)
            
        Returns:
        --------
        float
            The association measure s(w, A, B)
        """
        if target_word not in self.embeddings:
            return np.nan
            
         # Filter out words not in vocabulary
        A_in_vocab = [a for a in A if a in self.embeddings]
        B_in_vocab = [b for b in B if b in self.embeddings]
        
        if not A_in_vocab or not B_in_vocab:
            return np.nan
            
        # Calculate mean cosine similarity with attribute set A
        a_cos_sim = np.mean([self.cosine_similarity(target_word, a) for a in A_in_vocab])
        
        # Calculate mean cosine similarity with attribute set B
        b_cos_sim = np.mean([self.cosine_similarity(target_word, b) for b in B_in_vocab])
        
        # Return the difference (association measure)
        return a_cos_sim - b_cos_sim
    
    def test_statistic(self, X: List[str], Y: List[str], A: List[str], B: List[str]) -> float:
        """
        Calculate the test statistic S(X, Y, A, B) as defined in WEAT methodology
        S(X, Y, A, B) = sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)
        
        Parameters:
        -----------
        X, Y : List[str]
            Lists of target words
        A, B : List[str]
            Lists of attribute words (e.g., male and female terms)
            
        Returns:
        --------
        float
            The test statistic S(X, Y, A, B)
        """
        # Filter out words not in vocabulary
        X_in_vocab = [x for x in X if x in self.embeddings]
        Y_in_vocab = [y for y in Y if y in self.embeddings]
        
        if not X_in_vocab or not Y_in_vocab:
            return np.nan
            
        # Calculate association measures for words in X
        x_assoc = np.sum([self.association_measure(x, A, B) for x in X_in_vocab])
        
        # Calculate association measures for words in Y
        y_assoc = np.sum([self.association_measure(y, A, B) for y in Y_in_vocab])
        
        # Return the difference (test statistic)
        return x_assoc - y_assoc
    
    def permutation_test(self, X: List[str], Y: List[str], A: List[str], B: List[str], n_permutations: int = 1000) -> Tuple[float, float]:
        """
        Perform a permutation test to calculate the p-value and effect size for WEAT
        
        Parameters:
        -----------
        X, Y : List[str]
            Lists of target words
        A, B : List[str]
            Lists of attribute words
        n_permutations : int
            Number of permutations to use (default: 1000)
            
        Returns:
        --------
        Tuple[float, float]
            p-value, effect size
        """
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Filter words not in vocabulary
        X_in_vocab = [x for x in X if x in self.embeddings]
        Y_in_vocab = [y for y in Y if y in self.embeddings]
        
        # Skip if not enough words are available
        if len(X_in_vocab) < 2 or len(Y_in_vocab) < 2:
            return np.nan, np.nan
        
        # Calculate the observed test statistic
        observed_stat = self.test_statistic(X_in_vocab, Y_in_vocab, A, B)
        
        # Calculate the association measures for all words in X and Y
        X_assoc = [self.association_measure(x, A, B) for x in X_in_vocab]
        Y_assoc = [self.association_measure(y, A, B) for y in Y_in_vocab]
        
        # Calculate the pooled standard deviation using Cohen's formula
        X_std = np.std(X_assoc, ddof=1)  # Sample standard deviation for X
        Y_std = np.std(Y_assoc, ddof=1)  # Sample standard deviation for Y
        
        # Using the simpler Cohen formula: sqrt((SD_X^2 + SD_Y^2)/2)
        pooled_std = np.sqrt((X_std**2 + Y_std**2) / 2)
        
        # Calculate effect size using Cohen's d
        if pooled_std > 0:
            effect_size = (np.mean(X_assoc) - np.mean(Y_assoc)) / pooled_std
        else:
            effect_size = np.nan
        
        # For efficiency, use a random sample of permutations instead of all possible permutations
        all_words = X_in_vocab + Y_in_vocab
        
        # Track permutation statistics
        perm_stats = []
        
        for _ in range(n_permutations):
            # Randomly partition words into new X and Y groups of equal original sizes
            np.random.shuffle(all_words)
            X_i = all_words[:len(X_in_vocab)]
            Y_i = all_words[len(X_in_vocab):]
            
            # Calculate test statistic for this permutation
            perm_stat = self.test_statistic(X_i, Y_i, A, B)
            perm_stats.append(perm_stat)
        
        # Calculate one-sided p-value (probability that random partition has larger test statistic)
        p_value = np.mean([stat > observed_stat for stat in perm_stats])
        
        return p_value, effect_size
    
    def analyse_all_categories(self, n_permutations: int = 1000) -> Dict:
        """
        Analyse all bias categories defined in the target_words dictionary using WEAT
        
        Parameters:
        -----------
        n_permutations : int
            Number of permutations for each test (default: 1000)
            
        Returns:
        --------
        Dict
            Dictionary of results containing p-values, effect sizes
        """
        results = {}
        
        for category, word_lists in target_words.items():
            print(f"Analysing category: {category}")
            X, Y = word_lists['X'], word_lists['Y']
            A, B = attribute_words['M'], attribute_words['F']  # A=male, B=female in our context
            
            p_value, effect_size = self.permutation_test(X, Y, A, B, n_permutations)
            
            results[category] = {
                'p_value': p_value,
                'effect_size': effect_size,
                'X_words': X,
                'Y_words': Y,
                'X_in_vocab': [x for x in X if x in self.embeddings],
                'Y_in_vocab': [y for y in Y if y in self.embeddings],
                'interpretation': self._get_interpretation(effect_size, p_value)
            }
            
            # Print result summary
            print(f"  P-value: {p_value:.4f}, Effect size: {effect_size:.4f}")
            print(f"  X words in vocab: {len(results[category]['X_in_vocab'])}/{len(X)}")
            print(f"  Y words in vocab: {len(results[category]['Y_in_vocab'])}/{len(Y)}")
            print(f"  Interpretation: {results[category]['interpretation']}")
            
            # Track the p-values and effect sizes
            if not np.isnan(p_value):
                self.p_values[category] = p_value
                self.effect_sizes[category] = effect_size
        
        self.results = results
        return results
    
    def _get_interpretation(self, effect_size: float, p_value: float) -> str:
        """
        Get a human-readable interpretation of WEAT results
        
        Parameters:
        -----------
        effect_size : float
            The effect size (standardized measure)
        p_value : float
            The p-value from permutation test
            
        Returns:
        --------
        str
            Human-readable interpretation
        """
        if np.isnan(effect_size) or np.isnan(p_value):
            return "Insufficient vocabulary coverage to determine bias"
            
        significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
        
        if effect_size > 0:
            direction = "Category X words are more associated with male attributes"
            return f"{direction} (Effect size {effect_size}, {significance}, p={p_value:.4f})"
        else:
            direction = "Category Y words are more associated with male attributes"
            return f"{direction} (Effect size {effect_size}, {significance}, p={p_value:.4f})"
        
    def generate_table_results(self) -> pd.DataFrame:
        """
        Generate a DataFrame of results suitable for publication
        
        Returns:
        --------
        pd.DataFrame
            Table of results
        """
        if not self.results:
            print("No results available. Run analyse_all_categories() first.")
            return None
        
        # Create a DataFrame with the results
        results_data = []
        
        for category, result in self.results.items():
            results_data.append({
                'Category': category,
                'X-Y Description': category.split(': ')[1] if ': ' in category else category,
                'Effect Size': result['effect_size'],
                'p-value': result['p_value'],
                'Significant': 'Yes' if result['p_value'] < 0.05 else 'No',
                'X Words in Vocab': f"{len(result['X_in_vocab'])}/{len(result['X_words'])}",
                'Y Words in Vocab': f"{len(result['Y_in_vocab'])}/{len(result['Y_words'])}"
            })
        
        df = pd.DataFrame(results_data)
        
        # Sort by significance and effect size
        df = df.sort_values(['Significant', 'Effect Size'], ascending=[False, False])
        
        return df
    
    def visualize_permutation_tests(self, 
        categories: List[str] = None,
        n_permutations: int = 1000,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
        kde: bool = True,
        hist: bool = True,
        significance_level: float = 0.05):
        """
        Create a visualization of permutation test distributions for selected categories.
        
        Parameters:
        -----------
        save_path : str,
            Path to save the figure
        categories : List[str], optional
            List of categories to visualize. If None, uses all categories in target_words
        n_permutations : int, optional
            Number of permutations to perform (default: 1000)
        figsize : Tuple[int, int], optional
            Figure size
        kde : bool, optional
            Whether to show kernel density estimate curve
        hist : bool, optional
            Whether to show histogram
        significance_level : float, optional
            Alpha level for significance (default: 0.05)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Get categories from target_words if not specified
        if categories is None:
            categories = list(target_words.keys())
        
        # Determine grid layout
        n_categories = len(categories)
        n_cols = min(3, n_categories)
        n_rows = (n_categories + n_cols - 1) // n_cols
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        # Track the maximum y-value for axis scaling
        max_density = 0
        
        # Process each category
        for i, category in enumerate(categories):
            # Get row and column for subplot
            row = i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Get word lists for this category
            X = target_words[category]['X']
            Y = target_words[category]['Y']
            A = attribute_words['M']
            B = attribute_words['F']
            
            # Filter words not in vocabulary
            X_in_vocab = [x for x in X if x in self.embeddings]
            Y_in_vocab = [y for y in Y if y in self.embeddings]
            
            # Calculate the observed test statistic
            observed_stat = self.test_statistic(X_in_vocab, Y_in_vocab, A, B)
            
            # Generate null distribution through permutation
            all_words = X_in_vocab + Y_in_vocab
            n_X = len(X_in_vocab)
            
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Perform permutations and collect test statistics
            perm_stats = []
            for _ in range(n_permutations):
                np.random.shuffle(all_words)
                X_i = all_words[:n_X]
                Y_i = all_words[n_X:]
                perm_stat = self.test_statistic(X_i, Y_i, A, B)
                perm_stats.append(perm_stat)
            
            # Calculate p-value
            p_value = np.mean([stat > observed_stat for stat in perm_stats])
            
            # Create the histogram/kde plot
            if hist and kde:
                sns.histplot(perm_stats, kde=True, ax=ax, color='gray', alpha=0.6, stat='density')
            elif hist:
                sns.histplot(perm_stats, kde=False, ax=ax, color='gray', alpha=0.6, stat='density')
            elif kde:
                sns.kdeplot(perm_stats, ax=ax, color='gray', fill=True, alpha=0.6)
            
            # Add vertical line for observed statistic
            if p_value < significance_level:
                line_color = 'red'
                significant = "significant"
            else:
                line_color = 'blue'
                significant = "not significant"
                
            ax.axvline(x=observed_stat, color=line_color, linestyle='-', linewidth=2, 
                    label=f'Observed statistic\n(p={p_value:.3f}, {significant})')
            
            # Update plot title and labels
            ax.set_title(f'{category}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Test Statistic', fontsize=12)
            if col == 0:  # Only add y-label for leftmost plots
                ax.set_ylabel('Density', fontsize=12)
            
            # Add legend
            ax.legend(loc='upper right', frameon=True, fontsize=8)
            
            # Track max density for consistent y-axis
            if ax.get_ylim()[1] > max_density:
                max_density = ax.get_ylim()[1]
        
        # Set consistent y-axis limits
        for i in range(n_categories):
            ax = fig.axes[i]
            ax.set_ylim(0, max_density * 1.1)
        
        # Add data description at bottom
        n_permutations_text = f'Based on {n_permutations} permutations per category'
        fig.text(0.5, 0.01, n_permutations_text, ha='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.02, 1, 0.92])
        
        # Save
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        
        plt.close()
        
        return
    
# ===== SWEAT analysis =====
class GenderBiasSWEATAnalyser:
    """
    Implements the Sliced Word Embedding Association Test (SWEAT) for comparing gender bias
    across different corpora (e.g., French Rap vs. General French).
    """
    
    def __init__(self, corpus1_embeddings, corpus2_embeddings, corpus1_name="Corpus1", corpus2_name="Corpus2"):
        """
        Initialize the SWEAT analyzer with word embeddings from two different corpora.
        
        Parameters:
        -----------
        corpus1_embeddings : dict or KeyedVectors or Word2Vec
            Word embeddings from the first corpus (e.g., French Rap)
        corpus2_embeddings : dict or KeyedVectors or Word2Vec
            Word embeddings from the second corpus (e.g., FrWac or French Wikipedia)
        corpus1_name : str
            Name of the first corpus (for display purposes)
        corpus2_name : str
            Name of the second corpus (for display purposes)
        """
        self.corpus1_embeddings = self._process_embeddings(corpus1_embeddings)
        self.corpus2_embeddings = self._process_embeddings(corpus2_embeddings)
        self.corpus1_name = corpus1_name
        self.corpus2_name = corpus2_name
        self.results = {}
        
    def _process_embeddings(self, embeddings):
        """
        Process different input embedding formats into a standard format.
        
        Parameters:
        -----------
        embeddings : dict or KeyedVectors or Word2Vec
            Word embeddings in various formats
            
        Returns:
        --------
        dict
            Dictionary mapping words to numpy vectors
        """
        # If embeddings is a Word2Vec model, extract its KeyedVectors
        if hasattr(embeddings, 'wv'):
            embeddings = embeddings.wv
            
        # If embeddings is a KeyedVectors object, convert to dict
        if hasattr(embeddings, 'key_to_index') and hasattr(embeddings, 'get_vector'):
            embedding_dict = {}
            for word in embeddings.key_to_index:
                embedding_dict[word] = embeddings.get_vector(word)
            return embedding_dict
            
        # If embeddings is already a dict mapping words to vectors, use it directly
        if isinstance(embeddings, dict):
            return embeddings
            
        raise TypeError("Embeddings must be a dict mapping words to vectors, "
                      "a gensim KeyedVectors object, or a Word2Vec model")
    
    def cosine_similarity(self, word1: str, word2: str, corpus_embeddings: dict) -> float:
        """
        Calculate the cosine similarity between two words in a specific corpus.
        
        Parameters:
        -----------
        word1, word2 : str
            Words to compare
        corpus_embeddings : dict
            Dictionary of word embeddings from the corpus
            
        Returns:
        --------
        float
            Cosine similarity between the word vectors
        """
        if word1 not in corpus_embeddings or word2 not in corpus_embeddings:
            return np.nan
            
        vec1 = corpus_embeddings[word1]
        vec2 = corpus_embeddings[word2]
        
        # Calculate cosine similarity manually
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return np.nan
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def association_measure(self, 
                          word: str, 
                          A: List[str], 
                          B: List[str], 
                          corpus_embeddings: dict) -> float:
        """
        Calculate the association measure s(w, A, B, D) for a word in a specific corpus.
        
        Parameters:
        -----------
        word : str
            Target word
        A, B : List[str]
            Lists of attribute words (e.g., male and female terms)
        corpus_embeddings : dict
            Dictionary of word embeddings from the corpus
            
        Returns:
        --------
        float
            The association measure s(w, A, B, D)
        """
        if word not in corpus_embeddings:
            return np.nan
            
        # Filter out words not in vocabulary
        A_in_vocab = [a for a in A if a in corpus_embeddings]
        B_in_vocab = [b for b in B if b in corpus_embeddings]
        
        if not A_in_vocab or not B_in_vocab:
            return np.nan
            
        # Calculate mean cosine similarity with attribute set A
        a_cos_sim = np.mean([self.cosine_similarity(word, a, corpus_embeddings) for a in A_in_vocab])
        
        # Calculate mean cosine similarity with attribute set B
        b_cos_sim = np.mean([self.cosine_similarity(word, b, corpus_embeddings) for b in B_in_vocab])
        
        # Return the difference (association measure)
        return a_cos_sim - b_cos_sim
    
    def sweat_score(self, 
                    W: List[str], 
                    A: List[str], 
                    B: List[str]) -> Tuple[float, float, float]:
        """
        Calculate the SWEAT score comparing bias between two corpora.
        
        Parameters:
        -----------
        W : List[str]
            List of target words to analyze
        A, B : List[str]
            Lists of attribute words (e.g., male and female terms)
            
        Returns:
        --------
        Tuple[float, float, float]
            SWEAT score, effect size, and p-value
        """
        # Filter for words that are in both corpora
        W_in_both = [w for w in W if w in self.corpus1_embeddings and w in self.corpus2_embeddings]
        
        if len(W_in_both) < 2:
            return np.nan, np.nan, np.nan
        
        # Calculate association measures for each word in both corpora
        corpus1_assocs = [self.association_measure(w, A, B, self.corpus1_embeddings) for w in W_in_both]
        corpus2_assocs = [self.association_measure(w, A, B, self.corpus2_embeddings) for w in W_in_both]
        
        # Remove any NaN values that might have occurred
        valid_indices = [i for i, (a1, a2) in enumerate(zip(corpus1_assocs, corpus2_assocs)) 
                        if not (np.isnan(a1) or np.isnan(a2))]
        
        if len(valid_indices) < 2:
            return np.nan, np.nan, np.nan
            
        valid_corpus1_assocs = [corpus1_assocs[i] for i in valid_indices]
        valid_corpus2_assocs = [corpus2_assocs[i] for i in valid_indices]
        valid_words = [W_in_both[i] for i in valid_indices]
        
        # Calculate the SWEAT score: sum of differences in association measures
        sweat_score = sum(valid_corpus1_assocs) - sum(valid_corpus2_assocs)
        
        # Calculate effect size using pooled standard deviation (Cohen's d)
        diff_assocs = np.array(valid_corpus1_assocs) - np.array(valid_corpus2_assocs)
        effect_size = np.mean(diff_assocs) / np.std(diff_assocs, ddof=1) if len(diff_assocs) > 1 else np.nan
        
        # Compute statistical significance through permutation test
        p_value = self._permutation_test(valid_corpus1_assocs, valid_corpus2_assocs)
        
        return sweat_score, effect_size, p_value
    
    def _permutation_test(self, assoc1: List[float], assoc2: List[float], n_permutations: int = 1000) -> float:
        """
        Perform a permutation test to calculate the p-value for SWEAT.
        
        Parameters:
        -----------
        assoc1, assoc2 : List[float]
            Association measures for the same words in two corpora
        n_permutations : int
            Number of permutations to use (default: 1000)
            
        Returns:
        --------
        float
            p-value
        """
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Calculate observed difference
        observed_diff = sum(assoc1) - sum(assoc2)
        
        # Combine all values for permutation
        all_values = assoc1 + assoc2
        n = len(assoc1)
        
        # Perform permutations
        perm_diffs = []
        for _ in range(n_permutations):
            # Shuffle the values
            np.random.shuffle(all_values)
            # Split into new groups
            perm_assoc1 = all_values[:n]
            perm_assoc2 = all_values[n:]
            # Calculate difference
            perm_diff = sum(perm_assoc1) - sum(perm_assoc2)
            perm_diffs.append(perm_diff)
        
        # Calculate two-sided p-value
        # (because we're interested in whether the corpora differ, not just in one direction)
        p_value = np.mean([abs(diff) >= abs(observed_diff) for diff in perm_diffs])
        
        return p_value
    
    def analyze_category(self, 
                       category: str,
                       target_words: Dict[str, Dict[str, List[str]]],
                       attribute_words: Dict[str, List[str]]) -> Dict:
        """
        Analyze a specific category using SWEAT.
        
        Parameters:
        -----------
        category : str
            Category name (e.g., 'B1', 'B2', etc.)
        target_words : Dict
            Dictionary containing target word sets
        attribute_words : Dict
            Dictionary containing attribute word sets
            
        Returns:
        --------
        Dict
            Dictionary with SWEAT results
        """
        # Extract word sets
        X = target_words[category]['X']
        Y = target_words[category]['Y']
        M = attribute_words['M']
        F = attribute_words['F']
        
        # Calculate SWEAT scores for both X and Y word sets
        x_sweat, x_effect, x_pvalue = self.sweat_score(X, M, F)
        y_sweat, y_effect, y_pvalue = self.sweat_score(Y, M, F)
        
        # Calculate vocab coverage
        x_in_corpus1 = [x for x in X if x in self.corpus1_embeddings]
        x_in_corpus2 = [x for x in X if x in self.corpus2_embeddings]
        x_in_both = [x for x in X if x in self.corpus1_embeddings and x in self.corpus2_embeddings]
        
        y_in_corpus1 = [y for y in Y if y in self.corpus1_embeddings]
        y_in_corpus2 = [y for y in Y if y in self.corpus2_embeddings]
        y_in_both = [y for y in Y if y in self.corpus1_embeddings and y in self.corpus2_embeddings]
        
        # Store results
        result = {
            'X_SWEAT_score': x_sweat,
            'X_effect_size': x_effect,
            'X_p_value': x_pvalue,
            'Y_SWEAT_score': y_sweat,
            'Y_effect_size': y_effect,
            'Y_p_value': y_pvalue,
            'X_in_corpus1': x_in_corpus1,
            'X_in_corpus2': x_in_corpus2,
            'X_in_both': x_in_both,
            'Y_in_corpus1': y_in_corpus1,
            'Y_in_corpus2': y_in_corpus2,
            'Y_in_both': y_in_both,
            'interpretation': self._get_interpretation(x_sweat, x_pvalue, y_sweat, y_pvalue)
        }
        
        return result
    
    def analyze_all_categories(self, 
                             target_words: Dict[str, Dict[str, List[str]]],
                             attribute_words: Dict[str, List[str]]) -> Dict:
        """
        Analyze all categories using SWEAT.
        
        Parameters:
        -----------
        target_words : Dict
            Dictionary containing target word sets
        attribute_words : Dict
            Dictionary containing attribute word sets
            
        Returns:
        --------
        Dict
            Dictionary with SWEAT results for all categories
        """
        results = {}
        
        for category in target_words.keys():
            print(f"Analyzing category: {category}")
            result = self.analyze_category(category, target_words, attribute_words)
            results[category] = result
            
            # Print result summary
            print(f"  X: SWEAT score = {result['X_SWEAT_score']:.4f}, Effect size = {result['X_effect_size']:.4f}, p-value = {result['X_p_value']:.4f}")
            print(f"  Y: SWEAT score = {result['Y_SWEAT_score']:.4f}, Effect size = {result['Y_effect_size']:.4f}, p-value = {result['Y_p_value']:.4f}")
            print(f"  X words in both corpora: {len(result['X_in_both'])}/{len(target_words[category]['X'])}")
            print(f"  Y words in both corpora: {len(result['Y_in_both'])}/{len(target_words[category]['Y'])}")
            print(f"  Interpretation: {result['interpretation']}")
            
        self.results = results
        return results
    
    def _get_interpretation(self, 
                          x_sweat: float, 
                          x_pvalue: float, 
                          y_sweat: float, 
                          y_pvalue: float) -> str:
        """
        Get a human-readable interpretation of SWEAT results.
        
        Parameters:
        -----------
        x_sweat, y_sweat : float
            SWEAT scores for X and Y word sets
        x_pvalue, y_pvalue : float
            p-values for X and Y word sets
            
        Returns:
        --------
        str
            Human-readable interpretation
        """
        if np.isnan(x_sweat) or np.isnan(y_sweat):
            return "Insufficient vocabulary coverage to determine difference in bias"
            
        x_sig = "significant" if x_pvalue < 0.05 else "not significant"
        y_sig = "significant" if y_pvalue < 0.05 else "not significant"
        
        x_direction = "more male-associated" if x_sweat > 0 else "more female-associated"
        y_direction = "more male-associated" if y_sweat > 0 else "more female-associated"
        
        interpretation = f"X words are {x_direction} in {self.corpus1_name} compared to {self.corpus2_name} ({x_sig}, p={x_pvalue:.4f}). "
        interpretation += f"Y words are {y_direction} in {self.corpus1_name} compared to {self.corpus2_name} ({y_sig}, p={y_pvalue:.4f})."
        
        return interpretation
    
    def generate_table_results(self) -> pd.DataFrame:
        """
        Generate a DataFrame of results suitable for publication.
        
        Returns:
        --------
        pd.DataFrame
            Table of results
        """
        if not self.results:
            print("No results available. Run analyze_all_categories() first.")
            return None
        
        rows = []
        
        for category, result in self.results.items():
            # For X words
            rows.append({
                'Category': category,
                'Word Set': 'X',
                'Description': category.split(': ')[1] if ': ' in category else category,
                'SWEAT Score': result['X_SWEAT_score'],
                'Effect Size': result['X_effect_size'],
                'p-value': result['X_p_value'],
                'Significant': 'Yes' if result['X_p_value'] < 0.05 else 'No',
                f'Words in {self.corpus1_name}': len(result['X_in_corpus1']),
                f'Words in {self.corpus2_name}': len(result['X_in_corpus2']),
                'Words in Both': len(result['X_in_both'])
            })
            
            # For Y words
            rows.append({
                'Category': category,
                'Word Set': 'Y',
                'Description': category.split(': ')[1] if ': ' in category else category,
                'SWEAT Score': result['Y_SWEAT_score'],
                'Effect Size': result['Y_effect_size'],
                'p-value': result['Y_p_value'],
                'Significant': 'Yes' if result['Y_p_value'] < 0.05 else 'No',
                f'Words in {self.corpus1_name}': len(result['Y_in_corpus1']),
                f'Words in {self.corpus2_name}': len(result['Y_in_corpus2']),
                'Words in Both': len(result['Y_in_both'])
            })
        
        df = pd.DataFrame(rows)
        
        # Sort by significance and effect size
        df = df.sort_values(['Significant', 'Effect Size'], ascending=[False, False])
        
        return df
    
    def visualize_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize SWEAT results comparing bias across corpora.
        
        Parameters:
        -----------
        save_path : str, optional
            If provided, save the figure to this path
            
        Returns:
        --------
        plt.Figure
            The figure object
        """
        if not self.results:
            print("No results available. Run analyze_all_categories() first.")
            return None
        
        # Extract categories and effect sizes
        categories = list(self.results.keys())
        x_effects = [self.results[cat]['X_effect_size'] for cat in categories]
        y_effects = [self.results[cat]['Y_effect_size'] for cat in categories]
        x_pvalues = [self.results[cat]['X_p_value'] for cat in categories]
        y_pvalues = [self.results[cat]['Y_p_value'] for cat in categories]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set width of bars
        bar_width = 0.35
        x_pos = np.arange(len(categories))
        
        # Create bars with different colors based on significance
        x_colors = ['#3182bd' if p < 0.05 else '#9ecae1' for p in x_pvalues]
        y_colors = ['#e6550d' if p < 0.05 else '#fdae6b' for p in y_pvalues]
        
        # Create bars
        ax.bar(x_pos - bar_width/2, x_effects, bar_width, color=x_colors, label=f'X words ({self.corpus1_name} vs {self.corpus2_name})')
        ax.bar(x_pos + bar_width/2, y_effects, bar_width, color=y_colors, label=f'Y words ({self.corpus1_name} vs {self.corpus2_name})')
        
        # Add labels and title
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Effect Size (Cohen\'s d)', fontsize=12)
        ax.set_title(f'SWEAT Effect Sizes: {self.corpus1_name} vs {self.corpus2_name}', fontsize=14, fontweight='bold')
        
        # Add category labels on x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        # Add reference lines for effect size interpretation
        for y, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
            ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.axhline(y=-y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.text(len(categories)-0.5, y, label, ha='right', va='bottom', color='gray', fontsize=8)
            ax.text(len(categories)-0.5, -y, label, ha='right', va='top', color='gray', fontsize=8)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1,color='#3182bd', label=f'X words, significant (p < 0.05)'),
            plt.Rectangle((0,0),1,1,color='#9ecae1', label=f'X words, not significant'),
            plt.Rectangle((0,0),1,1,color='#e6550d', label=f'Y words, significant (p < 0.05)'),
            plt.Rectangle((0,0),1,1,color='#fdae6b', label=f'Y words, not significant')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Add interpretation text
        plt.figtext(0.5, 0.01, 
                   "Positive values: Words have stronger male associations in " + self.corpus1_name + " compared to " + self.corpus2_name + "\n"
                   "Negative values: Words have stronger female associations in " + self.corpus1_name + " compared to " + self.corpus2_name, 
                   ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            
        return fig
    
    def word_level_analysis(self, 
                         category: str, 
                         word_set: str, 
                         target_words: Dict[str, Dict[str, List[str]]],
                         attribute_words: Dict[str, List[str]],
                         top_n: int = 10,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Perform and visualize word-level analysis for a specific category and word set.
        
        Parameters:
        -----------
        category : str
            Category name (e.g., 'B1', 'B2', etc.)
        word_set : str
            Word set to analyze ('X' or 'Y')
        target_words : Dict
            Dictionary containing target word sets
        attribute_words : Dict
            Dictionary containing attribute word sets
        top_n : int
            Number of top words to display (default: 10)
        save_path : str, optional
            If provided, save the figure to this path
            
        Returns:
        --------
        plt.Figure
            The figure object
        """
        # Extract word sets
        words = target_words[category][word_set]
        M = attribute_words['M']
        F = attribute_words['F']
        
        # Find words that are in both corpora
        words_in_both = [w for w in words if w in self.corpus1_embeddings and w in self.corpus2_embeddings]
        
        if len(words_in_both) < 2:
            print(f"Not enough words in both corpora for category {category}, word set {word_set}")
            return None
        
        # Calculate association measures for each word in both corpora
        word_data = []
        for word in words_in_both:
            corpus1_assoc = self.association_measure(word, M, F, self.corpus1_embeddings)
            corpus2_assoc = self.association_measure(word, M, F, self.corpus2_embeddings)
            
            if not (np.isnan(corpus1_assoc) or np.isnan(corpus2_assoc)):
                diff = corpus1_assoc - corpus2_assoc
                word_data.append((word, corpus1_assoc, corpus2_assoc, diff))
        
        # Sort by absolute difference
        word_data.sort(key=lambda x: abs(x[3]), reverse=True)
        
        # Take top_n words
        top_words = word_data[:top_n]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Word-level differences
        words = [item[0] for item in top_words]
        diffs = [item[3] for item in top_words]
        colors = ['#3182bd' if d > 0 else '#e6550d' for d in diffs]
        
        y_pos = np.arange(len(words))
        ax1.barh(y_pos, diffs, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(words)
        ax1.set_xlabel('Difference in Association (Corpus1 - Corpus2)')
        ax1.set_title(f'Top {top_n} Words with Largest Association Differences')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 2: Scatter plot of associations in both corpora
        corpus1_assocs = [item[1] for item in word_data]
        corpus2_assocs = [item[2] for item in word_data]
        
        ax2.scatter(corpus1_assocs, corpus2_assocs, alpha=0.7)
        
        # Add word labels for top words
        for word, c1, c2, _ in top_words:
            ax2.annotate(word, (c1, c2), fontsize=9)
        
        # Add diagonal line
        min_val = min(min(corpus1_assocs), min(corpus2_assocs))
        max_val = max(max(corpus1_assocs), max(corpus2_assocs))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax2.set_xlabel(f'Association in {self.corpus1_name}')
        ax2.set_ylabel(f'Association in {self.corpus2_name}')
        ax2.set_title('Association Measures Across Corpora')
        
        # Add quadrant labels
        ax2.text(0.9*max_val, 0.9*max_val, 'Male in both', ha='right')
        ax2.text(0.9*min_val, 0.9*max_val, 'Female in Corpus1\nMale in Corpus2', ha='left')
        ax2.text(0.9*max_val, 0.9*min_val, 'Male in Corpus1\nFemale in Corpus2', ha='right')
        ax2.text(0.9*min_val, 0.9*min_val, 'Female in both', ha='left')
        
        # Add horizontal and vertical lines at 0
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle(f'Word-Level Analysis for Category {category}, {word_set} Words', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            
        return fig