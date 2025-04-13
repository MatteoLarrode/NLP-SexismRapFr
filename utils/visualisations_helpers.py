# ===============================================
# === Helper functions for data visualisation ===
# ===============================================
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
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
