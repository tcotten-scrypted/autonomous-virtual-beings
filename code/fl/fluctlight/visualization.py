"""
Visualization utilities for Fluctlight model analysis.

This module provides functions for generating and saving visualization plots
for model test results and training metrics.
"""

import os
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_test_results(
    csv_file: str,
    output_dir: Optional[str] = None,
    save_plots: bool = True
) -> None:
    """
    Generate visualization plots for test results.
    
    Args:
        csv_file: Path to CSV file containing test results
        output_dir: Directory to save plots (default: tmp/visualizations)
        save_plots: Whether to save plots to files (default: True)
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.join("tmp", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and process data
    df = pd.read_csv(csv_file, names=['Input', 'Expected', 'Actual', 'Errors', 'RMSE'])
    df['RMSE'] = pd.to_numeric(df['RMSE'])
    df['Errors'] = pd.to_numeric(df['Errors'])
    
    # Set style - using a built-in style that's guaranteed to exist
    plt.style.use('default')
    sns.set_theme()  # This will apply seaborn's styling
    
    # 1. Heatmap of RMSE error distribution
    plt.figure(figsize=(10, 6))
    heatmap_data = df.pivot_table(index="Input", values="RMSE", aggfunc="mean")
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Error Distribution (RMSE per Input Pattern)")
    plt.ylabel("Input Pattern")
    plt.xlabel("Test Cases")
    if save_plots:
        plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    plt.close()
    
    # 2. Scatter plot of expected vs. actual output
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df["Expected"], 
        y=df["Actual"], 
        hue=df["Errors"], 
        palette="coolwarm", 
        edgecolor="gray"
    )
    plt.plot(df["Expected"], df["Expected"], "r--", label="Ideal Prediction Line")
    plt.xlabel("Expected Output")
    plt.ylabel("Predicted Output")
    plt.title("Expected vs. Predicted Outputs")
    plt.legend()
    if save_plots:
        plt.savefig(os.path.join(output_dir, "prediction_scatter.png"))
    plt.close()
    
    # 3. Line plot of sequence evolution
    df_sorted = df.sort_values("Errors")
    plt.figure(figsize=(10, 5))
    plt.plot(
        df_sorted.index, 
        df_sorted["RMSE"], 
        marker="o", 
        linestyle="-", 
        color="blue", 
        label="RMSE Trend"
    )
    plt.xlabel("Test Cases")
    plt.ylabel("RMSE")
    plt.title("Error Evolution Across Test Cases")
    plt.legend()
    plt.grid()
    if save_plots:
        plt.savefig(os.path.join(output_dir, "error_evolution.png"))
    plt.close()
    
    # 4. Summary statistics plot
    plt.figure(figsize=(8, 6))
    stats = {
        'Total Tests': len(df),
        'Passed': len(df[df['Errors'] == 0]),
        'Failed': len(df[df['Errors'] > 0]),
        'Avg RMSE': df['RMSE'].mean(),
        'Max RMSE': df['RMSE'].max(),
        'Min RMSE': df['RMSE'].min()
    }
    plt.bar(stats.keys(), stats.values())
    plt.xticks(rotation=45)
    plt.title("Test Results Summary")
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, "summary_stats.png"))
    plt.close()
