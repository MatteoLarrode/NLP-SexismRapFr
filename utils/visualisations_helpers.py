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
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
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

def plot_gender_bias(df, figsize=(12, 8), marker_size=100, p_threshold=0.05, 
                    significant_marker='*', show_legend=True, save_path=None):
    """
    Create a plot visualizing gender bias effect sizes across categories with significance markers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing gender bias analysis results with p-values and effect sizes
    figsize : tuple, optional
        Figure size (width, height) in inches.
    marker_size : int, optional
        Base size for markers.
    p_threshold : float, optional
        P-value threshold for significance indication. Default is 0.05.
    significant_marker : str, optional
        Marker to use for significant results.
    show_legend : bool, optional
        Whether to show the legend.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object with the plot.
    """
    # Create a copy of the dataframe
    raw_df = df.copy()
    
    # Extract model names from the index
    model_names = raw_df.index.tolist()
    
    # Get bias categories
    bias_categories = raw_df.columns.tolist()
    
    # Create a more detailed mapping for plot title
    category_descriptions = {
        'B1_career_family': 'Male-Career, Female-Family',
        'B2_mathsci_arts': 'Male-Math/Science, Female-Arts',
        'B3_intel_appearance': 'Male-Intelligence, Female-Appearance',
        'B4_strength_weakness': 'Male-Strength, Female-Weakness',
        'B5_status_love': 'Male-Status, Female-Love'
    }
    
    # Extract data for plotting
    plot_data = []
    
    for model in model_names:
        for category in bias_categories:
            cell_value = raw_df.loc[model, category]
            
            # Extract p-value and effect size using regex
            p_match = re.search(r"'p_value':\s*([\d.]+)", str(cell_value))
            es_match = re.search(r"'effect_size':\s*([-\d.]+)", str(cell_value))
            
            if p_match and es_match:
                p_value = float(p_match.group(1))
                effect_size = float(es_match.group(1))
                
                # Add to plot data
                plot_data.append({
                    'Model': model,
                    'Category': category,
                    'Effect_Size': effect_size,
                    'P_Value': p_value,
                    'Significant': p_value < p_threshold
                })
    
    # Convert to DataFrame for easier plotting
    plot_df = pd.DataFrame(plot_data)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up colors for different models
    model_colors = sns.color_palette("Set2", len(model_names))
    color_map = {model: model_colors[i] for i, model in enumerate(model_names)}
    
    # Set up markers for different models
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd']
    marker_map = {model: markers[i % len(markers)] for i, model in enumerate(model_names)}

    # First create a mapping to uniformly distribute points across each column
    position_map = {}
    
    # For each category, calculate positions
    for category in bias_categories:
        # Get all models that have data for this category
        models_in_category = []
        for model in model_names:
            model_data = plot_df[(plot_df['Model'] == model) & (plot_df['Category'] == category)]
            if not model_data.empty:
                models_in_category.append(model)
        
        # Calculate positions spread across the column
        column_width = 0.8  # Width of the spread within column
        n_models = len(models_in_category)
        
        for i, model in enumerate(models_in_category):
            # If only one model, center it
            if n_models == 1:
                offset = 0
            # Otherwise, distribute evenly
            else:
                offset = column_width * (i / (n_models - 1) - 0.5)
            
            # Store the position
            position_map[(category, model)] = offset
    
    # Now plot using the pre-calculated positions
    for model in model_names:
        model_data = plot_df[plot_df['Model'] == model]
        
        for idx, row in model_data.iterrows():
            category = row['Category']
            effect_size = row['Effect_Size']
            significant = row['Significant']
            
            # Get the pre-calculated position offset
            if (category, model) in position_map:
                offset = position_map[(category, model)]
            else:
                offset = 0  # Fallback if no position calculated
            
            # Calculate final x-position
            x_pos = bias_categories.index(category) + 1
            x_jitter = x_pos + offset
            
            # Plot point - add label only once per model
            if category == model_data['Category'].iloc[0]:
                ax.scatter(x_jitter, effect_size, 
                         color=color_map[model], 
                         marker=marker_map[model],
                         s=marker_size,
                         alpha=0.8,
                         label=model)
            else:
                ax.scatter(x_jitter, effect_size, 
                         color=color_map[model], 
                         marker=marker_map[model],
                         s=marker_size,
                         alpha=0.8)
            
            # Add significance marker if significant
            if significant:
                ax.scatter(x_jitter, effect_size, 
                         marker=significant_marker, 
                         s=marker_size/2, 
                         color='black')
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add vertical lines to separate categories
    for i in range(len(bias_categories) - 1):
        ax.axvline(x=i + 1.5, color='black', linestyle='--', alpha=0.3)
    
    # Set the x-axis ticks and labels
    ax.set_xticks(range(1, len(bias_categories) + 1))
    ax.set_xticklabels([category_descriptions[cat] for cat in bias_categories], fontsize=10)
    
    # Set the labels (no title)
    ax.set_ylabel('WEAT Effect Size', fontsize=12)
    
    # Add note about significance marker
    ax.text(0.02, 0.02, f"{significant_marker} = p < {p_threshold}", transform=ax.transAxes)
    
    # Add legend only for model markers if requested
    if show_legend:
        # Get current handles and labels
        handles, labels = ax.get_legend_handles_labels()
        
        # Place the legend inside the figure at the top
        legend = ax.legend(handles, labels, 
                         title="Models", 
                         loc='upper center', 
                         bbox_to_anchor=(0.5, -0.05),
                         ncol=3,  # Adjust the number of columns as needed
                         frameon=True,
                         fontsize=9)
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path is not None:
        # Create directories if they don't exist
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig