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


# ===== WEAT experiment =====
# Define the attribute and target word lists for French rap
attribute_words = {
    'M': ['homme', 'mec', 'gars', 'frère', 'il', 'lui', 'son', 'fils', 'père', 'oncle', 'grand-père', 'mâle', 'king', 'bro', 'kho'],
    'F': ['femme', 'meuf', 'fille', 'sœur', 'elle', 'sa', 'fille', 'mère', 'tante', 'grand-mère', 'gazelle', 'go', 'miss', 'bae']
}

target_words = {
    'B1_career_family': {
        'X': ['business', 'patron', 'patronne', 'money', 'travail', 'boss', 'cash', 'hustle', 'bureau', 'carrière'],
        'Y': ['foyer', 'parents', 'maison', 'enfants', 'famille', 'mariage', 'domestique']
    },
    'B2_mathsci_arts': {
        'X': ['calcul', 'logique', 'science', 'chiffres', 'physique', 'maths', 'chimie'],
        'Y': ['poésie', 'art', 'danse', 'littérature', 'chanson', 'peinture']
    },
    'B3_intel_appearance': {
        'X': ['brillant','brillante', 'intelligent', 'intelligente', 'stratège', 'cerveau', 'sage', 'lucide', 'génie'],
        'Y': ['beau', 'belle', 'mince', 'moche', 'laid', 'laide', 'joli', 'jolie', 'maigre', 'gros', 'grosse', 'corps']
    },
    'B4_strength_weakness': {
        'X': ['confiant','confiante', 'puissant', 'puissante', 'force', 'dominat', 'dominante', 'fort', 'forte'],
        'Y': ['faible', 'fragile', 'timide' , 'doux' ,'douce', 'sensible', 'soumis', 'soumise', 'peur', 'vulnérable']
    },
    'B5_status_love': {
        'X': ['oseille', 'thune', 'francs', 'euros', 'dollars', 'bijoux', 'marques', 'luxe', 'rolex', 'chaine', 'fric'],
        'Y': ['amour', 'sentiments', 'cœur', 'passion', 'fidèle', 'romantique', 'relation', 'aimer', 'émotions', 'attachement']
    }
}

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
            
            # Add shaded area for p-value region
            if observed_stat > np.median(perm_stats):  # Right-tailed test
                x_fill = np.linspace(observed_stat, max(perm_stats) * 1.1, 100)
                try:
                    y_fill = ax.get_lines()[0].get_ydata()[-len(x_fill):]
                    ax.fill_between(x_fill, y_fill, alpha=0.2, color=line_color)
                except:
                    # Fallback if KDE line isn't available
                    pass
            else:  # Left-tailed test
                x_fill = np.linspace(min(perm_stats) * 1.1, observed_stat, 100)
                try:
                    y_fill = ax.get_lines()[0].get_ydata()[:len(x_fill)]
                    ax.fill_between(x_fill, y_fill, alpha=0.2, color=line_color)
                except:
                    # Fallback if KDE line isn't available
                    pass
            
            # Update plot title and labels
            ax.set_title(f'{category}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Test Statistic', fontsize=10)
            if col == 0:  # Only add y-label for leftmost plots
                ax.set_ylabel('Density', fontsize=10)
            
            # Add legend
            ax.legend(loc='upper right', frameon=True, fontsize=8)
            
            # Track max density for consistent y-axis
            if ax.get_ylim()[1] > max_density:
                max_density = ax.get_ylim()[1]
        
        # Set consistent y-axis limits
        for i in range(n_categories):
            ax = fig.axes[i]
            ax.set_ylim(0, max_density * 1.1)
        
        # Add overall title
        plt.suptitle('Permutation Test Distributions by Category', fontsize=16, fontweight='bold', y=0.98)
        
        # Add explanatory subtitle
        fig.text(0.5, 0.94, 
                'Distribution of test statistics under the null hypothesis with observed values shown as vertical lines.\n'
                'Red lines indicate statistically significant results (p < 0.05).',
                ha='center', fontsize=12)
        
        # Add data description at bottom
        n_permutations_text = f'Based on {n_permutations} permutations per category'
        fig.text(0.5, 0.01, n_permutations_text, ha='center', fontsize=10)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.02, 1, 0.92])
        
        # Save
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        
        plt.close()
        
        return