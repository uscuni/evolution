import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm

def pivot_tbl(df,lvl):
    grouped = (
        df
        .assign(weighted_count=lambda df: df['num_buildings'])
        .groupby(["year_mode", f"level_{lvl}_label"])['weighted_count']
        .sum()
        .reset_index()
    )
    pivot_df = grouped.pivot(index="year_mode", columns=f"level_{lvl}_label", values="weighted_count").fillna(0)
    pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0)
    return pivot_pct

def pivot_tbl_cum(df,lvl):
    # Group and sum num_buildings per year_mode and type
    grouped = (
        df
        .groupby(["year_mode", f"level_{lvl}_label"])['num_buildings']
        .sum()
        .reset_index()
    )
    
    # Pivot to have types as columns
    pivot_df = grouped.pivot(
        index="year_mode",
        columns=f"level_{lvl}_label",
        values='num_buildings'
    ).fillna(0)
    
    # Cumulative sum along the year_mode axis
    pivot_cum = pivot_df.cumsum()
    
    # Convert to cumulative proportions (optional)
    pivot_cum_pct = pivot_cum.div(pivot_cum.sum(axis=1), axis=0)
    
    return pivot_cum_pct


def line_plot(table):
    n_categories = len(table.columns)
    cmap = cm.get_cmap("tab20b", n_categories)
    colors = [cmap(i) for i in range(n_categories)]
    
    plt.figure(figsize=(12, 6))
    
    for i, col in enumerate(table.columns):
        plt.plot(
            table.index.astype(str),  # convert intervals to string for x-axis
            table[col],
            label=str(col),
            color=colors[i],
            marker='o'
        )
    
    plt.xlabel("Year Mode")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

def hist (table): 
    n_categories = len(table.columns)
    cmap = cm.get_cmap("tab20b", n_categories)  # can also try "tab20c", "tab20", "hsv"
    colors = [cmap(i) for i in range(n_categories)]
    table.plot(kind="bar", stacked=True, color=colors, figsize=(12,6))
    plt.legend(bbox_to_anchor=(1.05, 1))