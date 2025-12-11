import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm

taxonomy_colors = [
    "#4069BC",
    "#7CBAE4",
    "#E69C63",
    "#eec1d5",
    "#E0665F",
    "#ECBF43",
    "#b2cd32",
    "#1F943E",
]
taxonomy_cmap = {i: col for i, col in enumerate(taxonomy_colors, 1)}

names = {
    1: "Incoherent Large-Scale Homogeneous Fabric",
    2: "Incoherent Large-Scale Heterogeneous Fabric",
    3: "Incoherent Small-Scale Linear Fabric",
    4: "Incoherent Small-Scale Sparse Fabric",
    5: "Incoherent Small-Scale Compact Fabric",
    6: "Coherent Interconnected Fabric",
    7: "Coherent Dense Disjoint Fabric",
    8: "Coherent Dense Adjacent Fabric",
}

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
    f, ax = plt.subplots(figsize=(12, 6))
    n_categories = table.shape[1]
    if n_categories <= 8:
        color = taxonomy_cmap
    else:
        cmap = cm.get_cmap("tab20b", n_categories)
        color = [cmap(i) for i in range(n_categories)]
        
    for i, col in enumerate(table.columns):
        ax.plot(
            table.index.astype(str),  # convert intervals to string for x-axis
            table[col],
            label=str(col),
            color=color[i + 1] if n_categories <= 8 else color[i],
            marker='o'
        )
    
    ax.set_xlabel("Year Mode")
    ax.set_ylabel("Proportion")
    plt.xticks(rotation=90)
    ax.legend(bbox_to_anchor=(1.05, 1))
    handles, labels = ax.get_legend_handles_labels()
    if table.shape[1] == 8:
        legend = [names[int(label)] for label in labels]
    else:
        legend = labels
    ax.legend(handles, legend,
              bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    sns.despine()
    plt.tight_layout()
    

def hist(table): 
    f, ax = plt.subplots(figsize=(12, 6))
    if table.shape[1] <= 8:
        color = taxonomy_cmap
    else:
        cmap = plt.get_cmap("tab20b", table.shape[1])
        color = [cmap(i) for i in range(table.shape[1])]
    table.plot.bar(stacked=True, color=color, ax=ax, width=1)
    ax.legend(bbox_to_anchor=(1.05, 1))
    # Rename legend using type_names
    handles, labels = ax.get_legend_handles_labels()
    if table.shape[1] == 8:
        legend = [names[int(label)] for label in labels]
    else:
        legend = labels
    ax.legend(handles, legend,
              bbox_to_anchor=(1.01, .98), loc='upper left', frameon=False)
    sns.despine()