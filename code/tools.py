import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

taxonomy_colors = { 3:["#4069BC","#7CBAE4","#E69C63","#eec1d5","#E0665F","#ECBF43","#b2cd32","#1F943E"],
                    2:["#4069BC", "#eec1d5", "#ECBF43", "#b2cd32"],
                    1:["#4069BC", "#ECBF43"],
                    4:["#4069BC",
                       "#7CBAE4","#7CBAE4",
                       "#E69C63",
                       "#eec1d5","#eec1d5",
                       "#E0665F","#E0665F",
                       "#ECBF43","#ECBF43",
                       "#b2cd32","#b2cd32","#b2cd32","#b2cd32",
                       "#1F943E","#1F943E"]
}

level_4_mapping = {1:1,2:2,3:2,4:3,5:4,6:4,7:5,8:5,9:6,10:6,11:6,12:7,13:7,14:7,15:7,16:8,17:8}

names = {3:{
    1: "Incoherent Large-Scale Homogeneous Fabric",
    2: "Incoherent Large-Scale Heterogeneous Fabric",
    3: "Incoherent Small-Scale Linear Fabric",
    4: "Incoherent Small-Scale Sparse Fabric",
    5: "Incoherent Small-Scale Compact Fabric",
    6: "Coherent Interconnected Fabric",
    7: "Coherent Dense Disjoint Fabric",
    8: "Coherent Dense Adjacent Fabric"},
         2: {
    1: "Incoherent Large-Scale \nFabric",
    2: "Incoherent Small-Scale \nFabric",
    3: "Coherent Interconnected \nFabric",
    4: "Coherent Dense \nFabric"},
         1:{
    1: "Incoherent Fabric",
    2: "Coherent Fabric"}
    
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

def pivot_tbl_abs(df,lvl):
    grouped = (
        df
        .assign(weighted_count=lambda df: df['num_buildings'])
        .groupby(["year_mode", f"level_{lvl}_label"])['weighted_count']
        .sum()
        .reset_index()
    )
    pivot_df = grouped.pivot(index="year_mode", columns=f"level_{lvl}_label", values="weighted_count").fillna(0)
    return pivot_df

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
    

def hist(table, level): 
    f, ax = plt.subplots(figsize=(8, 6))
    
    # --- 1. Define Colors ---
    if level <= 3:
        colors = taxonomy_colors[level]
    elif level == 4:
        # CHANGED: Dynamically map level 4 columns to level 3 colors using the mapping
        colors = [taxonomy_colors[3][level_4_mapping[int(col)] - 1] for col in table.columns]
    else:
        cmap = plt.get_cmap("tab20b", table.shape[1])
        colors = [cmap(i) for i in range(table.shape[1])]

    # --- 2. Determine Hatches (for level 4 sub-categories) ---
    hatches_for_cols = []
    if level == 4:
        color_counts = {}
        # Hatch patterns for 2nd, 3rd, 4th, etc. sub-variants
        hatch_patterns = ['///', '....', 'xxx']
        
        for color in colors:
            count = color_counts.get(color, 0)
            if count == 0:
                hatches_for_cols.append('')  # First one gets no hatch
            else:
                # Apply a hatch pattern based on the sub-variant index
                hatch_idx = (count - 1) % len(hatch_patterns)
                hatches_for_cols.append(hatch_patterns[hatch_idx])
            color_counts[color] = count + 1
    else:
        # No hatches for other levels
        hatches_for_cols = [''] * table.shape[1]

    # --- 3. Plot the Chart ---
    # Add edgecolor to make hatches visible
    table.plot.bar(stacked=True, color=colors, ax=ax, width=1, edgecolor='black')

    # --- 4. Apply Hatches to Patches ---
    # Pandas stacks column by column. The first len(table) patches belong 
    # to column 0, the next len(table) patches to column 1, etc.
    num_rows = len(table)
    for i, patch in enumerate(ax.patches):
        col_idx = i // num_rows
        if col_idx < len(hatches_for_cols):
            hatch = hatches_for_cols[col_idx]
            patch.set_linewidth(0)
            if hatch:  # Only apply if the string is not empty
                patch.set_hatch(hatch)
                patch.set_edgecolor('white') 
                

    # --- 5. Custom Legend ---
    if level <= 4:
        legend_elements = []
        # Use level 3 names and colors for both level 3 and 4 legends
        for key, name in names[3].items():
            # key-1 because dictionary keys are 1-indexed, but lists are 0-indexed
            if (key - 1) < len(taxonomy_colors[3]): 
                col = taxonomy_colors[3][key - 1]
                legend_elements.append(mpatches.Patch(facecolor=col, label=name))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, .98), loc='upper left', frameon=False)
    else:
        ax.legend(bbox_to_anchor=(1.01, .98), loc='upper left', frameon=False)

    sns.despine()

def area_chart(table, level): 
    f, ax = plt.subplots(figsize=(10, 6))
    
    # --- 1. Define Colors and Transparencies ---
    if level <= 3:
        colors = taxonomy_colors[level]
    elif level == 4:
        # Pre-group level 4 labels by their level 3 parent to establish absolute order
        # This ensures e.g., 12=1st(100%), 13=2nd(80%), 14=3rd(60%), regardless of missing data
        l4_groups = {}
        for l4, l3 in level_4_mapping.items():
            l4_groups.setdefault(l3, []).append(l4)
        for l3 in l4_groups:
            l4_groups[l3].sort()
            
        colors = []
        for col in table.columns:
            col_int = int(col)
            l3_parent = level_4_mapping[col_int]
            hex_color = taxonomy_colors[3][l3_parent - 1]
            
            # Find the absolute index of this label in its group (0, 1, 2, etc.)
            alpha_idx = l4_groups[l3_parent].index(col_int)
            
            # Calculate alpha: 1st = 1.0, 2nd = 0.8, 3rd = 0.6... 
            # We use max(0.1, ...) just to ensure it never becomes completely invisible
            alpha_val = max(0.1, 1.0 - (0.2 * alpha_idx))
            
            # Convert hex to RGBA so pandas area plot can read the transparency
            rgba = mcolors.to_rgba(hex_color, alpha=alpha_val)
            colors.append(rgba)
            
    else:
        cmap = plt.get_cmap("tab20b", table.shape[1])
        colors = [cmap(i) for i in range(table.shape[1])]

    # --- 2. Plot the Stacked Area Chart ---
    # CHANGED: Changed to plot.area(). Linewidth=0 keeps it smooth without outlines.
    table = table.sort_index()
    table.plot.area(stacked=True, color=colors, ax=ax, linewidth=0)
    ax.get_legend().remove()


    sns.despine()
    ax.set_xticks(range(len(table.index)))
    clean_labels = [str(idx).replace('(', '').replace(']', '').replace('.0', '').replace(', ', ' - ') for idx in table.index]
    ax.set_xticklabels(clean_labels, rotation=90)