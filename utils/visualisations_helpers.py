# ===============================================
# === Helper functions for data visualisation ===
# ===============================================
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import re
import seaborn as sns
import pandas as pd
from scipy import stats
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from IPython.display import display, Markdown

def set_visualization_style():
    plt.style.use('seaborn-v0_8-colorblind')
    font_path = '/Users/matteolarrode/Library/Fonts/cmunss.ttf'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams.update({
        'text.usetex': False,
        #'font.family': 'serif',
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
        # Add this line to use ASCII hyphen instead of Unicode minus
        'axes.unicode_minus': False
    })

def display_similarity_results_table(df, sort_by='Pearson_Correlation', ascending=False, 
                                   highlight_best=True, precision=2):
    """
    Format and display similarity validation results as a markdown table.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing similarity validation results with columns:
        Model, Valid_Pairs, Total_Pairs, Coverage, Pearson_Correlation, 
        Pearson_P_Value, Spearman_Correlation, Spearman_P_Value
    sort_by : str, optional
        Column to sort the results by. Default is 'Pearson_Correlation'.
    ascending : bool, optional
        Sort order. Default is False (descending).
    highlight_best : bool, optional
        Whether to highlight the best value in each metric column.
    precision : int, optional
        Number of decimal places for floating point values.
        
    Returns:
    --------
    markdown_table : str
        Formatted markdown table string.
    """
    # Create a copy of the dataframe
    table_df = df.copy()
    
    # Sort the dataframe
    table_df = table_df.sort_values(by=sort_by, ascending=ascending)
    
    # Format the dataframe for display
    formatted_df = pd.DataFrame()
    formatted_df['Model'] = table_df['Model']
    
    # Format Pearson correlation with p-value
    formatted_df["Pearson's r (p-value)"] = table_df.apply(
        lambda row: f"{row['Pearson_Correlation']:.{precision}f} ({row['Pearson_P_Value']:.2e})",
        axis=1
    )
    
    # Format Spearman correlation with p-value
    formatted_df["Spearman's r (p-value)"] = table_df.apply(
        lambda row: f"{row['Spearman_Correlation']:.{precision}f} ({row['Spearman_P_Value']:.2e})",
        axis=1
    )
    
    # Format pairs with coverage percentage
    formatted_df['Pairs (coverage)'] = table_df.apply(
        lambda row: f"{int(row['Valid_Pairs'])}/{int(row['Total_Pairs'])} ({(row['Valid_Pairs']/row['Total_Pairs']*100):.1f}%)",
        axis=1
    )
    
    # Highlight the best values if requested
    if highlight_best:
        # Store original values for comparison
        pearson_values = table_df['Pearson_Correlation']
        spearman_values = table_df['Spearman_Correlation']
        coverage_values = table_df['Valid_Pairs'] / table_df['Total_Pairs']
        
        # Find index of best values
        best_pearson_idx = pearson_values.idxmax()
        best_spearman_idx = spearman_values.idxmax()
        best_coverage_idx = coverage_values.idxmax()
        
        # Apply bold formatting to cells with best values
        pearson_col = "Pearson's r (p-value)"
        spearman_col = "Spearman's r (p-value)"
        pairs_col = "Pairs (coverage)"
        
        formatted_df.loc[best_pearson_idx, pearson_col] = f"**{formatted_df.loc[best_pearson_idx, pearson_col]}**"
        formatted_df.loc[best_spearman_idx, spearman_col] = f"**{formatted_df.loc[best_spearman_idx, spearman_col]}**"    
    
    # Convert to markdown
    markdown_table = formatted_df.to_markdown(index=False)
    
    # Return the markdown table
    return markdown_table

def display_analogy_category_table(df, top_n_models=None, precision=1, 
                                  sort_by='Overall_Accuracy', ascending=False,
                                  selected_categories=None):
    """
    Create a table showing performance of models across different analogy categories.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing analogy test results with columns:
        Model, Overall_Accuracy, Skipped_Questions, plus category columns
    top_n_models : int, optional
        Number of top models to display. If None, shows all models.
    precision : int, optional
        Number of decimal places for floating point values.
    sort_by : str, optional
        Column to sort the models by. Default is 'Overall_Accuracy'.
    ascending : bool, optional
        Sort order. Default is False (descending).
    selected_categories : list, optional
        List of specific categories to include. If None, includes all categories.
        
    Returns:
    --------
    markdown_table : str
        Formatted markdown table string.
    """
    # Create a copy of the dataframe
    table_df = df.copy()
    
    # Identify category columns (all columns except Model, Overall_Accuracy, Skipped_Questions)
    all_category_cols = [col for col in table_df.columns 
                       if col not in ['Model', 'Overall_Accuracy', 'Skipped_Questions']]
    
    # Filter categories if specified
    if selected_categories is not None:
        category_cols = [col for col in all_category_cols if col in selected_categories]
    else:
        category_cols = all_category_cols
    
    # Sort by specified column
    table_df = table_df.sort_values(by=sort_by, ascending=ascending)
    
    # Limit to top N models if specified
    if top_n_models is not None:
        table_df = table_df.head(top_n_models)
    
    # Create a new dataframe for display
    formatted_df = pd.DataFrame()
    
    # Set Model as the first column
    formatted_df['Model'] = table_df['Model']
    
    # Add Overall_Accuracy
    formatted_df['Overall'] = (table_df['Overall_Accuracy'] * 100).round(precision).astype(str) + '%'
    
    # Add category columns with pretty names
    for col in category_cols:
        # Create a nicer column name
        pretty_name = col.replace('-', ' ').replace('_', ' ').title()
        
        # Format the values
        formatted_df[pretty_name] = table_df[col].apply(
            lambda x: f"{x * 100:.{precision}f}%" if not pd.isna(x) else "N/A"
        )
    
    # Find and highlight the best model for each category
    for col in category_cols:
        pretty_name = col.replace('-', ' ').replace('_', ' ').title()
        
        # Skip if column doesn't exist (shouldn't happen, but just in case)
        if pretty_name not in formatted_df.columns:
            continue
            
        # Get values for this category
        category_values = table_df[col]
        
        # Skip if all values are NaN
        if category_values.isna().all():
            continue
            
        # Find the best model for this category
        best_idx = category_values.idxmax(skipna=True)
        
        # Check if the best value is not NaN
        if not pd.isna(table_df.loc[best_idx, col]):
            # Get the current formatted value
            current_value = formatted_df.loc[best_idx, pretty_name]
            # Add bold formatting
            formatted_df.loc[best_idx, pretty_name] = f"**{current_value}**"
    
    # Find and highlight the best overall model
    best_overall_idx = table_df['Overall_Accuracy'].idxmax(skipna=True)
    current_overall = formatted_df.loc[best_overall_idx, 'Overall']
    formatted_df.loc[best_overall_idx, 'Overall'] = f"**{current_overall}**"
    
    # Convert to markdown
    markdown_table = formatted_df.to_markdown(index=False)
    
    return markdown_table

def plot_gendered_average_similarity_by_model(similarity_df, save_path=None):
    """
    Plot average cosine similarity between gendered word pairs for each model.
    
    Parameters:
    -----------
    similarity_df : pd.DataFrame
        DataFrame with similarity scores (output from compare_gendered_word_similarities)
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        The figure object
    """
    set_visualization_style()
    
    plt.figure(figsize=(12, 6))
    
    # Calculate average similarity per model and sort by similarity
    model_avgs = similarity_df.groupby('model')['similarity'].mean().reset_index()
    model_avgs = model_avgs.sort_values('similarity', ascending=False)
    
    # Create the bar plot with sorted models
    ax = sns.barplot(x='model', y='similarity', data=model_avgs, order=model_avgs['model'])
    
    # Add data labels on top of bars
    for i, row in enumerate(model_avgs.itertuples()):
        ax.text(i, row.similarity + 0.02, f"{row.similarity:.3f}", 
                ha='center', va='bottom', fontsize=9)
    
    # Add reference line for typical threshold of 0.5
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='0.5 threshold')
    
    # Styling
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Average Cosine Similarity', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    
    return 

def plot_gendered_similarity_by_pair(similarity_df, top_n=None, save_path=None):
    """
    Plot cosine similarity for each gendered word pair across models.
    
    Parameters:
    -----------
    similarity_df : pd.DataFrame
        DataFrame with similarity scores (output from compare_gendered_word_similarities)
    top_n : int, optional
        If provided, show only the top N pairs with highest variation across models
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        The figure object
    """
    set_visualization_style()
    
    # Calculate statistics for each pair and sort
    pair_stats = similarity_df.groupby('pair')['similarity'].agg(['mean', 'std']).reset_index()
    
    # Sort by std first (if looking at top_n variations) or by mean (for overall view)
    if top_n is not None:
        pair_stats = pair_stats.sort_values('std', ascending=False)
        sort_column = 'std'
        sort_label = "highest variation"
    else:
        pair_stats = pair_stats.sort_values('mean', ascending=False)
        sort_column = 'mean'
        sort_label = "highest similarity"
    
    # Select pairs to show
    if top_n is not None and top_n < len(pair_stats):
        pairs_to_show = pair_stats.head(top_n)['pair'].tolist()
        plot_data = similarity_df[similarity_df['pair'].isin(pairs_to_show)]
    else:
        pairs_to_show = pair_stats['pair'].tolist()
        plot_data = similarity_df
    
    # Determine figure size based on number of pairs
    n_pairs = len(pairs_to_show)
    fig_width = max(10, min(18, n_pairs * 1.2))
    
    plt.figure(figsize=(fig_width, 8))
    
    # Create bar plot for each pair across models, ordered by the statistics
    ax = sns.barplot(x='pair', y='similarity', hue='model', data=plot_data, 
                    order=pairs_to_show)
    
    # Styling
    plt.xlabel('Gendered Word Pair', fontsize=14)
    plt.ylabel('Cosine Similarity', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add reference line for typical threshold of 0.5
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='0.5 threshold')
    
    # Adjust legend
    plt.legend(title='Model', loc='upper right', frameon=True, facecolor='white', edgecolor='black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return

def display_gender_bias_compact_table(df, models=None, highlight_significant=True, 
                                     p_threshold=0.05, precision=3):
    """
    Format and display gender bias analysis results in a compact table format
    showing p-values and effect sizes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing gender bias analysis results with p-values and effect sizes
        for different categories (B1_career_family, B2_mathsci_arts, etc.)
    models : list, optional
        List of specific models to include. If None, includes all models.
    highlight_significant : bool, optional
        Whether to highlight statistically significant results.
    p_threshold : float, optional
        P-value threshold for significance highlighting. Default is 0.05.
    precision : int, optional
        Number of decimal places for floating point values.
        
    Returns:
    --------
    markdown_table : str
        Formatted markdown table string.
    """
    # Create a copy of the dataframe
    raw_df = df.copy()
    
    # Extract model names from the index
    model_names = raw_df.index.tolist()
    
    # Filter models if specified
    if models is not None:
        model_names = [m for m in model_names if m in models]
    
    # Get bias categories
    bias_categories = raw_df.columns.tolist()
    
    # Create a new dataframe for the formatted table
    formatted_df = pd.DataFrame()
    formatted_df['Model'] = model_names
    
    # Create category name mapping
    category_names = {
        'B1_career_family': 'Career-Family',
        'B2_mathsci_arts': 'Math/Science-Arts',
        'B3_intel_appearance': 'Intelligence-Appearance',
        'B4_strength_weakness': 'Strength-Weakness',
        'B5_status_love': 'Status-Love'
    }
    
    # Extract p-values and effect sizes for each category
    for category in bias_categories:
        pretty_name = category_names.get(category, category.replace('_', ' '))
        
        for model in model_names:
            cell_value = raw_df.loc[model, category]
            
            # Extract p-value and effect size using regex
            p_match = re.search(r"'p_value':\s*([\d.]+)", str(cell_value))
            es_match = re.search(r"'effect_size':\s*([-\d.]+)", str(cell_value))
            
            if p_match and es_match:
                p_value = float(p_match.group(1))
                effect_size = float(es_match.group(1))
                
                # Format as effect size (p-value)
                combined = f"ES={effect_size:.{precision}f}, p={p_value:.{precision}f}"
                
                # Highlight significant results if requested
                if highlight_significant and p_value < p_threshold:
                    combined = f"**{combined}**"
                
                # Set the value in the dataframe
                if pretty_name not in formatted_df.columns:
                    formatted_df[pretty_name] = None
                
                formatted_df.loc[formatted_df['Model'] == model, pretty_name] = combined
    
    # Convert to markdown
    markdown_table = formatted_df.to_markdown(index=False)
    
    return markdown_table

def plot_gender_bias(df, figsize=(12, 8), marker_size=100, show_legend=True, save_path=None):
    """
    Plot gender bias effect sizes across categories with corpus/dim/algo encoding.
    """
    raw_df = df.copy()
    model_names = raw_df.index.tolist()
    bias_categories = raw_df.columns.tolist()

    category_descriptions = {
        'B1_career_family': 'Male-Career, Female-Family',
        'B2_mathsci_arts': 'Male-Math/Science, Female-Arts',
        'B3_intel_appearance': 'Male-Intelligence, Female-Appearance',
        'B4_strength_weakness': 'Male-Strength, Female-Weakness',
        'B5_status_love': 'Male-Status, Female-Love'
    }

    # Mappings
    corpus_color_map = {
        'frRap': '#4DA0E7',
        'frWiki': '#F3CA40',
        'frWac': '#5DD39E'
    }
    dim_marker_map = {
        '100': '^',
        '200': 'o',
        '1000': 's'
    }
    algo_edge_map = {
        'CBOW': 'black',
        'Skip-gram': 'none'
    }

    def extract_metadata(model_name):
        corpus = next((c for c in corpus_color_map if c in model_name), 'Other')
        dim_match = re.search(r'_(\d+)_', model_name)
        dim = dim_match.group(1) if dim_match else '200'
        algo = 'CBOW' if 'cbow' in model_name.lower() else 'Skip-gram'
        return corpus, dim, algo

    plot_data = []
    for model in model_names:
        for category in bias_categories:
            cell_value = raw_df.loc[model, category]
            es_match = re.search(r"'effect_size':\s*([-\d.]+)", str(cell_value))
            if es_match:
                effect_size = float(es_match.group(1))
                corpus, dim, algo = extract_metadata(model)
                plot_data.append({
                    'Model': model,
                    'Corpus': corpus,
                    'Dimension': dim,
                    'Algorithm': algo,
                    'Category': category,
                    'Effect_Size': effect_size,
                })

    plot_df = pd.DataFrame(plot_data)

    set_visualization_style()

    fig, ax = plt.subplots(figsize=figsize)

    position_map = {}
    for category in bias_categories:
        models_in_category = plot_df[plot_df['Category'] == category]['Model'].unique().tolist()
        column_width = 0.8
        n_models = len(models_in_category)
        for i, model in enumerate(models_in_category):
            offset = 0 if n_models == 1 else column_width * (i / (n_models - 1) - 0.5)
            position_map[(category, model)] = offset

    # Plotting
    set_visualization_style()

    legend_labels_seen = set()
    for _, row in plot_df.iterrows():
        category = row['Category']
        model = row['Model']
        x_base = bias_categories.index(category) + 1
        x_jitter = x_base + position_map.get((category, model), 0)
        y = row['Effect_Size']

        color = corpus_color_map[row['Corpus']]
        marker = dim_marker_map[row['Dimension']]
        edgecolor = algo_edge_map[row['Algorithm']]

        label_str = f"{row['Corpus']} {row['Dimension']}D {row['Algorithm']}"
        label = label_str if label_str not in legend_labels_seen else None
        if label: legend_labels_seen.add(label)

        ax.scatter(x_jitter, y,
                   color=color, marker=marker, s=marker_size,
                   edgecolor=edgecolor, linewidth=1.2,
                   label=label, alpha=0.9)

    # Styling
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    for i in range(len(bias_categories) - 1):
        ax.axvline(x=i + 1.5, color='black', linestyle='--', alpha=0.3)

    ax.set_xticks(range(1, len(bias_categories) + 1))
    ax.set_xticklabels([category_descriptions[cat] for cat in bias_categories], fontsize=12, rotation=15)

    ax.set_ylabel('WEAT Effect Size', fontsize=14)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        # Pair and sort legend entries: first by corpus, then dimension, then algorithm
        legend_tuples = list(zip(labels, handles))
        def legend_sort_key(x):
            label = x[0]
            parts = label.split()
            corpus_order = {'frRap': 0, 'frWiki': 1, 'frWac': 2}
            corpus = parts[0]
            dim = int(parts[1][:-1])  # e.g. '200D' -> 200
            algo = 0 if parts[2] == 'CBOW' else 1  # CBOW first
            return (corpus_order.get(corpus, 99), dim, algo)

        legend_tuples.sort(key=legend_sort_key)
        sorted_labels, sorted_handles = zip(*legend_tuples)

        legend = ax.legend(sorted_handles, sorted_labels, title="Models",
                           loc='upper center', bbox_to_anchor=(0.5, -0.2),
                           ncol=3, fontsize=11, title_fontsize=12,
                           frameon=True)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig