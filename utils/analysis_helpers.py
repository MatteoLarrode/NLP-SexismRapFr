# =========================================================
# === Helper functions for the word embeddings analysis ===
# =========================================================
import numpy as np
import itertools
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import List, Dict, Tuple

# ===== WEAT experiment =====
# Define the attribute and target word lists for French rap
attribute_words = {
    'M': ['homme', 'mec', 'gars', 'frère', 'il', 'lui', 'son', 'fils', 'père', 'oncle', 'grand-père', 'mâle', 'king', 'bro', 'kho'],
    'F': ['femme', 'meuf', 'fille', 'sœur', 'elle', 'sa', 'fille', 'mère', 'tante', 'grand-mère', 'gazelle', 'go', 'miss', 'bae']
}

target_words = {
    'B1_career_family': {
        'X_career': ['business', 'patron', 'patronne', 'money', 'travail', 'boss', 'cash', 'hustle', 'bureau', 'carrière'],
        'Y_family': ['foyer', 'parents', 'maison', 'enfants', 'famille', 'mariage', 'domestique']
    },
    'B2_mathsci_arts': {
        'X_mathsci': ['calcul', 'logique', 'science', 'chiffres', 'physique', 'maths', 'chimie'],
        'Y_arts': ['poésie', 'art', 'danse', 'littérature', 'chanson', 'peinture']
    },
    'B3_intel_appearance': {
        'X_intel': ['brillant','brillants', 'intelligent', 'intelligente', 'stratège', 'cerveau','sage', 'lucide', 'génie'],
        'Y_appearance': ['beau', 'belle', 'mince', 'moche', 'laid', 'laide', 'joli', 'jolie', 'maigre', 'gros', 'grosse', 'corps']
    },
    'B4_strength_weakness': {
        'X_strength': ['confiant','confiante', 'puissant', 'puissante', 'force', 'dominat', 'dominante', 'fort', 'forte'],
        'Y_weakness': ['faible', 'fragile', 'timide' , 'doux' ,'douce', 'sensible', 'soumis', 'soumise', 'peur', 'vulnérable']
    },
    'B5_status_love': {
        'X_status': ['oseille', 'thune', 'francs', 'euros', 'dollars', 'bijoux', 'marques', 'luxe', 'rolex', 'chaine', 'fric'],
        'Y_love': ['amour', 'sentiments', 'cœur', 'passion', 'fidèle', 'romantique', 'relation', 'aimer', 'émotions', 'attachement']
    }
}

class GenderBiasWEATAnalyser:
    def __init__(self, embeddings, embedding_dimension = None):
        """
        Initialize the analyzer with word embeddings
        
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
    
    def cosine_similarity(self, word1: str, word2: str) -> floar:
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
            p-value and effect size
        """
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
        
        # For efficiency, we'll use a random sample of permutations instead of all possible permutations
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
    
    def analyze_all_categories(self, n_permutations: int = 1000) -> Dict:
        """
        Analyze all bias categories defined in the target_words dictionary using WEAT
        
        Parameters:
        -----------
        n_permutations : int
            Number of permutations for each test (default: 1000)
            
        Returns:
        --------
        Dict
            Dictionary of results containing p-values and effect sizes
        """
        results = {}
        
        for category, word_lists in target_words.items():
            print(f"Analyzing category: {category}")
            X, Y = word_lists['X'], word_lists['Y']
            A, B = attribute_words['M'], attribute_words['F']  # A=male, B=female in our context
            
            p_value, effect_size = self.permutation_test(X, Y, A, B, n_permutations)
            
            results[category] = {
                'p_value': p_value,
                'effect_size': effect_size,
                'X_words': X,
                'Y_words': Y,
                'X_in_vocab': [x for x in X if x in self.model.wv],
                'Y_in_vocab': [y for y in Y if y in self.model.wv],
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
            strength = self._get_effect_strength(effect_size)
            return f"{direction} ({strength} effect, {significance}, p={p_value:.4f})"
        else:
            direction = "Category Y words are more associated with male attributes"
            strength = self._get_effect_strength(abs(effect_size))
            return f"{direction} ({strength} effect, {significance}, p={p_value:.4f})"
        
    def generate_table_results(self) -> pd.DataFrame:
        """
        Generate a DataFrame of results suitable for publication
        
        Returns:
        --------
        pd.DataFrame
            Table of results
        """
        if not self.results:
            print("No results available. Run analyze_all_categories() first.")
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
