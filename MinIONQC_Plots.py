#!/usr/bin/env python
"""
Created on Wed Jan 24 08:53:37 2024

@author: ian.michael.bollinger@gmail.com/researchconsultants@critical.consulting
"""
import os

import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import AutoMinorLocator
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
sns.set(style="whitegrid")

### TOOLS

def scientific_formatter(x, pos):
    return f"{x/1e7:.0f}x10$^{7}$" if x != 0 else "0"

def log10_major_ticks(min_val, max_val):
    """Generate major tick marks for a log10 scale."""
    major_ticks = np.logspace(np.floor(np.log10(min_val)), np.ceil(np.log10(max_val)), num=int(np.ceil(np.log10(max_val)) - np.floor(np.log10(min_val)) + 1))
    return major_ticks

def log10_minor_ticks(major_ticks):
    """Generate minor tick marks for a log10 scale."""
    minor_ticks = []
    for t in major_ticks[:-1]:
        minor_ticks.extend(np.linspace(t, t * 10, 10, endpoint=False))
    return minor_ticks

def calculate_N50(lengths):
    """
    Calculate the N50 value from a series of lengths.
    """
    if len(lengths) == 0:
        return 0  # or another appropriate value indicating no data

    sorted_lengths = sorted(lengths, reverse=True)
    cumsum_lengths = np.cumsum(sorted_lengths)
    total_length = cumsum_lengths[-1]
    return sorted_lengths[np.searchsorted(cumsum_lengths, total_length / 2)]

### PLOTTING

def plot_length_histogram(data, output_path, plot_format, plot_stat, p1m, q):
    """
    Plot a histogram of read lengths in stacked subplots.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(p1m * 960/75, p1m * 960/75))
    
    # Calculate logarithmically spaced bin edges
    min_length = max(data['sequence_length_template'].min(), 1)
    max_length = data['sequence_length_template'].max()
    bins = np.logspace(np.log10(min_length), np.log10(max_length), num=300)
       
    # Plot for 'All reads'
    sns.histplot(data[data['Q_cutoff'] == 'All reads']['sequence_length_template'], ax=axes[0], kde=False, bins=bins, color='#3b528bff')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Number of Reads', fontsize=p1m * 20)
    axes[0].set_title('Read Length Distribution - All Reads', fontsize=p1m * 22)
    axes[0].set_xlim(min_length, max_length)  # Set common x-axis range

    # Plot for 'Q>=7'
    sns.histplot(data[data['Q_cutoff'] == f'Q>={q}']['sequence_length_template'], ax=axes[1], kde=False, bins=bins, color='#5dc862ff')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Read Length (bases)', fontsize=p1m * 20)
    axes[1].set_ylabel('Number of Reads', fontsize=p1m * 20)
    axes[1].set_title(f'Read Length Distribution - Q>={q}', fontsize=p1m * 22)
    axes[1].set_xlim(min_length, max_length)  # Set common x-axis range

    # Set major and minor ticks for the x-axis
    major_ticks = log10_major_ticks(min_length, max_length)
    minor_ticks = log10_minor_ticks(major_ticks)
    for ax in axes:
        ax.set_xticks(major_ticks, minor=False)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth='0.5', color='grey')

    # Add the N50 lines and labels if applicable
    if plot_stat:
        for ax in axes:
            n50 = calculate_N50(data[data['Q_cutoff'] == ax.get_title().split(" - ")[-1]]['sequence_length_template'])
            ax.axvline(x=n50, color='black', linestyle='dashed')
            ax.text(x=n50, y=0.95*ax.get_ylim()[1], s=f'N50: {n50}', rotation=90, verticalalignment='top')

    # Add the title to the plot
    fig.suptitle('Sequence Length Histograms', fontsize=p1m * 25)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_path}/length_histogram.{plot_format}")
    plt.close()

def plot_qscore_histogram(data, output_path, plot_format, p1m, q):
    """
    Plot a histogram of Q scores in stacked subplots.
    """
    # Determine the common x-axis range
    min_qscore = min(data['mean_qscore_template'].min(), 0)  # Including 0 for safety
    max_qscore = data['mean_qscore_template'].max()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(p1m * 960/75, p1m * 960/75), sharex=True)
    
    # Plot for 'All reads'
    sns.histplot(data[data['Q_cutoff'] == 'All reads']['mean_qscore_template'], ax=axes[0], kde=False, bins=300, color='#3b528bff')
    axes[0].set_ylabel('Number of Reads', fontsize=p1m * 20)
    axes[0].set_title('Mean Quality (Q) Score Distribution - All Reads', fontsize=p1m * 22)
    axes[0].set_xlim(min_qscore, max_qscore)  # Set common x-axis range
    
    # Plot for 'Q>=7'
    sns.histplot(data[data['Q_cutoff'] == f'Q>={q}']['mean_qscore_template'], ax=axes[1], kde=False, bins=300, color='#5dc862ff')
    axes[1].set_xlabel('Mean Quality (Q) Score of Read', fontsize=p1m * 20)
    axes[1].set_ylabel('Number of Reads', fontsize=p1m * 20)
    axes[1].set_title(f'Mean Quality (Q) Score Distribution - Q>={q}', fontsize=p1m * 22)
    axes[1].set_xlim(min_qscore, max_qscore)  # Set common x-axis range
    
    # Apply minor ticks and grids to both histograms
    for ax in axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which='minor', axis='x', linestyle=':', linewidth='0.5', color='grey')
    
    # Add the title to the plot
    fig.suptitle('Sequence Quality (Q) Score Histograms', fontsize=p1m * 25)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_path}/qscore_histogram.{plot_format}")
    plt.close()

def plot_yield_over_time(data, output_path, muxes, plot_format, p1m, q):
    """
    Plot the yield over time.
    """
    # Creating the plot
    plt.figure(figsize=(p1m*960/75, p1m*480/75)) # plt.figure(figsize=(38.4, 19.2))
    palette = {"All reads": "#3b528bff", f"Q>={q}": "#5dc862ff"}
    sns.lineplot(data=data, x='hour', y='cumulative.bases.time', hue='Q_cutoff', palette=palette)

    plt.xlabel('Hours Into Run', size = p1m * 20)
    plt.ylabel('Total Yield in Gigabases (GB)', size = p1m * 20)
    plt.title('Gigabase (GB) Yield and Quality (Q) Over Time', size = p1m * 25)

    # Add the vertical lines for given intervals
    for interval in muxes:
        plt.axvline(x=interval, color='red', linestyle='dashed', alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_path}/yield_over_time.{plot_format}")
    plt.close()

def plot_yield_by_length(data, output_path, plot_format, p1m, q):
    """
    Plot the yield over time.
    """

    # Calculate xmax
    xmax = data.loc[data['cumulative.bases'] > 0.01 * data['cumulative.bases'].max(), 'sequence_length_template'].max()

    plt.figure(figsize=(p1m*960/75, p1m*480/75))
    sns.lineplot(data=data, x='sequence_length_template', y=data['cumulative.bases']/1e9, 
                 hue='Q_cutoff', palette="viridis")

    plt.xlabel('Minimum Read Length (bases)', size = p1m * 20)
    plt.ylabel('Total Yield in Gigabases (GB)', size = p1m * 20)
    plt.title('Read Length (bases) per Gigabase (GB) Generated', size = p1m * 25)
    plt.xlim(0, xmax)

    plt.tight_layout()
    plt.savefig(f"{output_path}/yield_by_length.{plot_format}")
    plt.close()

def plot_sequence_length_over_time(data, output_path, muxes, plot_format, p1m, q):
    # Create the plot
    plt.figure(figsize=(p1m*960/75, p1m*480/75))

    # Filter the data for 'All reads' and 'Q>={q}'
    df_all_reads = data[data['Q_cutoff'] == 'All reads']
    df_q = data[data['Q_cutoff'] == f'Q>={q}']

    # Get the count for each unique hour
    count_hours_all = df_all_reads['hour'].value_counts().to_dict()
    count_hours_q = df_q['hour'].value_counts().to_dict()

    # Create a trimmed dataframe with 'hour' values in the dictionary whose value is < 5
    df_all_reads_trimmed = df_all_reads[df_all_reads['hour'].map(count_hours_all) > 5]
    df_q_trimmed = df_q[df_q['hour'].map(count_hours_q) > 5]

    # Calculate lowess smoothed values with a smaller fraction for less smoothing
    lowess_all_reads = sm.nonparametric.lowess(df_all_reads_trimmed['sequence_length_template'], df_all_reads_trimmed['hour'], frac=0.25)
    lowess_q = sm.nonparametric.lowess(df_q_trimmed['sequence_length_template'], df_q_trimmed['hour'], frac=0.25)

    # Plotting the smoothed curves
    plt.plot(lowess_all_reads[:, 0], lowess_all_reads[:, 1], color='#3b528bff', lw=2, label='All Reads')
    plt.plot(lowess_q[:, 0], lowess_q[:, 1], color='#5dc862ff', lw=2, label=f'Q>={q}')

    # Trim the dataframes
    df_all_reads_trimmed = df_all_reads_trimmed[['hour', 'sequence_length_template']]
    df_q_trimmed = df_q_trimmed[['hour', 'sequence_length_template']]

    # Plotting with seaborn lineplot using trimmed dataframes
    sns.lineplot(data=df_all_reads_trimmed, x='hour', y='sequence_length_template', label='All Reads Mean', estimator='mean', color='#3b528bff', lw=1, linestyle='dashed')
    sns.lineplot(data=df_q_trimmed, x='hour', y='sequence_length_template', label=f'Q>={q} Mean', estimator='mean', color='#5dc862ff', lw=1, linestyle='dashed')

    # Set the plot labels and title
    plt.xlabel('Hours Into Run', size = p1m * 20)
    plt.ylabel('Mean Read Length (bases)', size = p1m * 20)
    plt.yticks(size = p1m * 7)
    plt.title('Sequence Length Over Time', size = p1m * 25)

    # Add the vertical lines for given intervals
    for interval in muxes:
        plt.axvline(x=interval, color='red', linestyle='dashed', alpha=0.5)

    # Adjust legend position and font size
    legend = plt.legend(title='Reads', loc='right', bbox_to_anchor=(1.225, 0.5))
    legend.get_title().set_fontsize(p1m * 12)
    for label in legend.get_texts():
        label.set_fontsize(p1m * 10)
        
    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_path}/length_by_hour.{plot_format}")
    plt.close()

def plot_qscore_over_time(data, output_path, muxes, plot_format, p1m, q):
    """
    Plot Q score over time
    """
    # Create the plot
    plt.figure(figsize=(p1m*960/75, p1m*480/75))

    # Filter out the negative values from 'sequence_length_template' in both subsets
    df_all_reads = data[(data['Q_cutoff'] == 'All reads') & (data['mean_qscore_template'] > 0)]
    df_q = data[(data['Q_cutoff'] == f'Q>={q}') & (data['mean_qscore_template'] > 0)]

    # Get the count for each unique hour
    count_hours_all = df_all_reads['hour'].value_counts().to_dict()
    count_hours_q = df_q['hour'].value_counts().to_dict()

    # Create a trimmed dataframe with 'hour' values in the dictionary whose value is < 5
    df_all_reads_trimmed = df_all_reads[df_all_reads['hour'].map(count_hours_all) > 5]
    df_q_trimmed = df_q[df_q['hour'].map(count_hours_q) > 5]

    # Calculate lowess smoothed values with a smaller fraction for less smoothing
    lowess_all_reads = sm.nonparametric.lowess(df_all_reads_trimmed['mean_qscore_template'], df_all_reads_trimmed['hour'], frac=0.25)
    lowess_q = sm.nonparametric.lowess(df_q_trimmed['mean_qscore_template'], df_q_trimmed['hour'], frac=0.25)

    # Plotting the smoothed curves
    plt.plot(lowess_all_reads[:, 0], lowess_all_reads[:, 1], color='#3b528bff', lw=2, label='All Reads')
    plt.plot(lowess_q[:, 0], lowess_q[:, 1], color='#5dc862ff', lw=2, label=f'Q>={q}')

    # Trim the dataframes
    df_all_reads_trimmed = df_all_reads_trimmed[['hour', 'mean_qscore_template']]
    df_q_trimmed = df_q_trimmed[['hour', 'mean_qscore_template']]

    # Plotting with seaborn lineplot using trimmed dataframes
    sns.lineplot(data=df_all_reads_trimmed, x='hour', y='mean_qscore_template', label='All Reads Mean', color='#3b528bff', lw=1, linestyle='dashed')
    sns.lineplot(data=df_q_trimmed, x='hour', y='mean_qscore_template', label=f'Q>={q} Mean', color='#5dc862ff', lw=1, linestyle='dashed')

    # Set the plot labels and title
    plt.xlabel('Hours Into Run', size = p1m * 20)
    plt.ylabel('Mean Quality (Q) Score', size = p1m * 20)
    plt.title('Quality (Q) Scores Over Time', size=p1m * 25)

    # Add the vertical lines for given intervals
    for interval in muxes:
        plt.axvline(x=interval, color='red', linestyle='dashed', alpha=0.5)

    # Adjust legend position and font size
    legend = plt.legend(title='Reads', loc='right', bbox_to_anchor=(1.225, 0.5))
    legend.get_title().set_fontsize(p1m * 12)
    for label in legend.get_texts():
        label.set_fontsize(p1m * 10)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_path}/q_by_hour.{plot_format}")
    plt.close()

def plot_reads_per_hour(data, output_path, muxes, plot_format, p1m, q):
    """
    Plot number of reads per hour
    """
    plt.figure(figsize=(p1m*960/75, p1m*480/75)) # plt.figure(figsize=(38.4, 19.2))
    palette = {"All reads": "#3b528bff", f"Q>={q}": "#5dc862ff"}
    min_x_value = int(data['hour'].min())  # Convert to integer
    max_x_value = int(data['hour'].max())  # Convert to integer
    
    # Ensure all hours are represented in the data
    all_hours = pd.DataFrame({'hour': range(min_x_value, max_x_value + 1)})
    data = pd.merge(all_hours, data, on='hour', how='left')
    data['reads_per_hour'].fillna(0, inplace=True)
    
    sns.pointplot(data=data, x='hour', y='reads_per_hour', hue='Q_cutoff', palette=palette)
    
    plt.xlabel('Hours Into Run')
    plt.ylabel('Number of Reads per Hour')
    plt.title('Reads Generated per Hour', size = p1m * 25)
    
    # Add the vertical lines for given intervals
    for interval in muxes:
        plt.axvline(x=interval, color='red', linestyle='dashed', alpha=0.5)
    
    plt.xlim(0, max_x_value*1.05)
    
    # Set X-ticks as the x_list
    plt.xticks(range(min_x_value, max_x_value + 1))
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/reads_per_hour.{plot_format}")
    plt.close()

# Function for channel_summary.png
def plot_channel_summary_histograms(df_all_reads, df_q_cutoff, output_dir, plot_format, p1m, q):
    # Rename the columns as per the R script
    rename_dict = {
        'total_bases': 'Number of Bases per Channel',
        'total_reads': 'Number of Reads per Channel',
        'mean_read_length': 'Mean Read Length per Channel',
        'median_read_length': 'Median Read Length per Channel'
    }
    df_all_reads['variable'] = df_all_reads['variable'].map(rename_dict).fillna(df_all_reads['variable'])
    df_q_cutoff['variable'] = df_q_cutoff['variable'].map(rename_dict).fillna(df_q_cutoff['variable'])

    # Specify the order of variables
    ordered_variables = [
        'Mean Read Length per Channel',
        'Median Read Length per Channel',
        'Number of Bases per Channel',
        'Number of Reads per Channel'
    ]

    # Create a 4x2 grid of subplots (4 variables, 2 dataframes)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(p1m*960/75, p1m*480/75), sharex='col')

    # Plot histograms and find the maximum value for y-axis
    max_value = 0
    for i, variable in enumerate(ordered_variables):
        # Plot for "All reads" dataframe
        df_var_all = df_all_reads[df_all_reads['variable'] == variable]
        ax_all = axes[0, i]
        sns.histplot(df_var_all['value'], ax=ax_all, color='#3b528bff', bins=30)
        max_value = max(max_value, ax_all.get_ylim()[1])

        # Plot for "Q-cutoff" dataframe
        df_var_cutoff = df_q_cutoff[df_q_cutoff['variable'] == variable]
        ax_cutoff = axes[1, i]
        sns.histplot(df_var_cutoff['value'], ax=ax_cutoff, color='#5dc862ff', bins=30)
        max_value = max(max_value, ax_cutoff.get_ylim()[1])
        
        if variable == "Number of Bases per Channel":
            ax_all.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
            ax_cutoff.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))

    # Adjust y-axis limits
    max_value *= 1.05
    for ax in axes.flatten():
        ax.set_ylim(0, max_value)

    # Set titles and labels
    for i, variable in enumerate(ordered_variables):
        axes[0, i].set_title(variable)
        axes[1, i].set_xlabel('')
        if i > 0:
            axes[0, i].set_ylabel("")
            axes[1, i].set_ylabel("")            
        else:
            axes[0, i].set_ylabel("Count")
            axes[1, i].set_ylabel("Count")

    # Add the title to the plot
    fig.suptitle('Flowcell Summary Histograms', fontsize=p1m * 25)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/channel_summary.{plot_format}', bbox_inches='tight')
    plt.close()

# Function for length_vs_q.png scatterplot
def plot_length_vs_q(data, output_dir, plot_format, p1m, q):
    # Filter for 'All reads'
    df_filtered = data[data['Q_cutoff'] == 'All reads']

    # Create figure and axes for the scatter plot
    fig, ax = plt.subplots(figsize=(p1m*960/75, p1m*960/75))

    # Check if the data is from MinION or PromethION
    if df_filtered['channel'].max() <= 512:
        # MinION
        # Define the normalization range based on 'events_per_base'
        norm = mcolors.LogNorm(vmin=df_filtered['events_per_base'].min(), vmax=df_filtered['events_per_base'].max())

        # Normalize 'events_per_base' values and apply the 'rocket' color map
        cmap = sns.color_palette("rocket", as_cmap=True)
        colors = cmap(norm(df_filtered['events_per_base']))

        # Create a scatter plot with normalized color values
        scatter = ax.scatter(df_filtered['sequence_length_template'], df_filtered['mean_qscore_template'],
                             color=colors, alpha=0.05, s=0.4)

        # Create colorbar in the figure, attached to the axes
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Events per Base (log scale)', rotation=270, labelpad=20)

    else:
        # PromethION
        # Use hexbin for 2D histogram with 'rocket' color map
        hexbin = ax.hexbin(df_filtered['sequence_length_template'], df_filtered['mean_qscore_template'], 
                           gridsize=50, cmap="rocket", bins='log')

        # Create colorbar in the figure, attached to the axes
        cbar = fig.colorbar(hexbin, ax=ax)
        cbar.set_label('Counts in bin', rotation=270, labelpad=20)

    # Set axis scales and labels
    ax.set_xscale('log')
    ax.set_xlabel('Read Length (bases)', size = p1m * 15)
    ax.set_ylabel('Mean Quality (Q) Score of Read', size = p1m * 15)
    ax.set_title('Read Length (bases) vs Quality (Q) Score', size = p1m * 25)
    
    # Set the y-axis to start at 0
    ax.set_ylim(bottom=0)
 
    # Set the x-axis to start at 0
    ax.set_xlim(left=1)

    # Enable minor gridlines on the x-axis
    ax.xaxis.grid(True, which='minor', linewidth=0.5)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f'{output_dir}/length_vs_q.{plot_format}')
    plt.close()

def plot_both_per_channel(df1, df2, output_dir, plot_format, p1m, q):
    # Function to process and pivot data
    def process_data(df):
        aggregated_df = df.groupby(['row', 'col']).agg({'value': 'mean'}).reset_index()
        return aggregated_df.pivot(index='row', columns='col', values='value')

    # Process both dataframes
    pivot_table1 = process_data(df1)
    pivot_table2 = process_data(df2)

    # Determine the common color scale for both heatmaps
    min_value = min(pivot_table1.min().min(), pivot_table2.min().min())
    max_value = max(pivot_table1.max().max(), pivot_table2.max().max())

    # Set up the plot with adjusted figsize and width ratios    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(p1m * 960/75, p1m * 960/75),  # Reduced width
                             gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.01}, sharey=True)

    # Plot heatmap for df1
    sns.heatmap(pivot_table1, annot=False, cmap="viridis", square=True, ax=axes[0], 
                vmin=min_value, vmax=max_value, cbar=False)
    axes[0].set_ylabel('Channel Row', fontsize=p1m * 25)

    # Plot heatmap for df2 with a colorbar
    cbar_ax = fig.add_axes([.93, .3, .02, .4])  # Position for the colorbar
    
    sns.heatmap(pivot_table2, annot=False, cmap="viridis", square=True, ax=axes[1], 
                vmin=min_value, vmax=max_value, cbar_ax=cbar_ax)
    
    # Calculate the positions based on min and max values
    tick_positions = [min_value + 1/3 * (max_value - min_value), 
                      min_value + 2/3 * (max_value - min_value)]
    cbar_ax.set_yticks(tick_positions)
    cbar_ax.set_yticklabels(['0.01', '0.02'])
    
    # Add title "GB/Channel" to Legend (Colorbar)
    cbar_ax.set_title("GB per\nChannel", fontsize=p1m * 15)

    # Set colorbar text size
    cbar_ax.tick_params(labelsize= p1m * 10)

    # Adjust tick label size and set ticks for both plots
    for ax in axes:
        ax.set_xlabel('')
        # Make sure there is no background grid
        ax.grid(False)
   
        # Find the number of rows and columns in the data
        nrows, ncols = pivot_table1.shape if ax == axes[0] else pivot_table2.shape
    
        # Set ticks at specific intervals
        # For x-axis, every 4 units; for y-axis, every 10 units
        xticks = np.arange(3.5, ncols, 4)
        yticks = np.arange(9.5, nrows, 10)
    
        ax.set_xticks(xticks, minor=False)
        ax.set_yticks(yticks, minor=False)
    
        # Setting new labels as simple integers
        ax.set_xticklabels([int(round(x,0)) for x in xticks], fontsize= p1m * 20)  # +1 if indexing starts from 1
        ax.set_yticklabels([int(round(y,0)) for y in yticks], fontsize= p1m * 20)
    
        ax.tick_params(axis='x', rotation=0, labelsize= p1m * 22)
        ax.tick_params(axis='y', labelsize= p1m * 20)
        
        # Draw a vertical white line from x = 1 to x = 15
        for x in range(1, 16):
            ax.axvline(x=x, color='white', linestyle='-', lw=2)

        # Draw a vertical white line from x = 1 to x = 15
        for y in range(1, 32):
            ax.axhline(y=y, color='white', linestyle='-', lw=2)

    # Set a centralized x-axis label for the entire figure
    fig.text(0.5, 0.04, 'Channel Column', ha='center', va='center', fontsize= p1m * 22)

    # Add light grey box behind each heatmap title
    for ax, title in zip(axes, ['All Reads', f'Q>={q}']):
        ax.text(0.5, 1.05, title, transform=ax.transAxes, fontsize= p1m * 22, 
                horizontalalignment='center', verticalalignment='center', 
                bbox=dict(facecolor='lightgrey', edgecolor='none', boxstyle='round,pad=0.5'))

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Add the title to the plot
    fig.suptitle('Gigabases (GB) per Channel Overview', fontsize=p1m * 25)        

    # # Save the dual plot with adjusted layout
    # plt.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(os.path.join(output_dir, f"gb_per_channel_overview.{plot_format}"))
    plt.close()

def plot_flowcell_overview(data, output_dir, plot_format, p1m, q):
    """
    Generate a flowcell overview plot

    Args:
    data (DataFrame): The DataFrame containing the data to plot.
    output_dir (str): Directory to save the output plot.
    plot_format (str, optional): Format of the saved plot. Defaults to 'png'.
    """
    # Create a large figure
    fig = plt.figure(figsize=(p1m*80, p1m*76.4)) # fig = plt.figure(figsize=(80, 76.4))
    
    # Define grid size
    num_columns = 16
    num_rows = 32

    # Add axes for subplots and colorbar
    grid_size = (num_rows, num_columns + 1)  # +1 to account for the colorbar
    axs = [plt.subplot2grid(grid_size, (i, j)) for i in range(num_rows) for j in range(num_columns)]

    # Filter for 'All reads' and create a copy to avoid SettingWithCopyWarning
    all_reads_data = data[data['Q_cutoff'] == 'All reads'].copy()
    all_reads_data['start_time_hours'] = all_reads_data['start_time'] / 3600  # Convert start_time to hours


    for i, ax in enumerate(axs):
        # Determine the row and column number for this subplot
        row_number = i // num_columns + 1
        col_number = i % num_columns + 1

        # Filter data for this specific subplot based on row and col
        subplot_data = all_reads_data[(all_reads_data['row'] == row_number) & 
                                      (all_reads_data['col'] == col_number)]
    
        if not subplot_data.empty and 'mean_qscore_template' in subplot_data.columns:
            valid_scores = subplot_data['mean_qscore_template'].dropna()
            if not valid_scores.empty:
                # Generate Mini plot
                mini_plot = sns.scatterplot(x='start_time_hours', y='sequence_length_template', 
                                hue='mean_qscore_template', data=subplot_data, ax=ax, 
                                palette="viridis", alpha=0.35)
    
                # Remove y-axis and x-axis labels of mini-plot if present
                ax.set_xlabel('')
                ax.set_ylabel('')              
          
                # Remove legend from mini_plot
                if ax.get_legend():
                    ax.get_legend().remove()
            else:
                logging.info(f"No valid 'mean_qscore_template' data for subplot {i}")
        else:
            logging.info(f"Empty data or missing 'mean_qscore_template' column for subplot {i}")

        # Set y-axis to log scale
        ax.set_yscale('log')

        # Set y-ticks and label size
        ax.set_yticks([1e+01, 1e+02, 1e+03, 1e+04, 1e+05])
        if i % num_columns != 0:
            ax.set_yticklabels([])
        else:
            ax.tick_params(axis='y', labelsize= p1m * 25)

        # Set x-ticks and label size
        ax.set_xticks([0, 10, 20, 30, 40])
        if i < num_columns * (num_rows - 1):
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', labelsize= p1m * 25)

    # Add overall Title, y-axis label, and x-axis label
    fig.text(0.5, 0.95, 'Individual Flowcell Read Length & Quality (Q) Over Time', ha='center', fontsize= p1m * 175)
    fig.text(0.05, 0.5, 'Read Length (bases)', va='center', rotation='vertical', fontsize= p1m * 150)
    fig.text(0.5, 0.05, 'Hours Into Run', ha='center', fontsize= p1m * 150)

    # Create a colorbar for the Viridis palette
    viridis = plt.get_cmap('viridis')
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=0, vmax=16))
    sm.set_array([])

    # Create a new axis for the colorbar with desired position
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_title('Q', size= p1m * 100)
    cbar.ax.tick_params(labelsize= p1m * 75)

    # Add light grey box behind each column heatmap title
    first_row_axes = axs[:num_columns]
    column_label_list = [str(i) for i in range(1, num_columns + 1)]
    for ax, title in zip(first_row_axes, column_label_list):
        ax.text(0.5, 1.7, title, transform=ax.transAxes, fontsize= p1m * 75, 
                horizontalalignment='center', verticalalignment='center', 
                bbox=dict(facecolor='lightgrey', edgecolor='none', boxstyle='round,pad=0.5'))

    # Calculate x position for the row labels (to the right of the subplots, but left of the colorbar)
    label_x_pos = (num_columns) / (num_columns + 1) * 0.9325  # Adjust this as needed
    
    # Create a list of integers for row titles
    row_label_list = [str(i) for i in range(1, num_rows + 1)]
    
    # Add row titles directly to the figure
    for i, ax in zip(range(num_rows), axs[::num_columns]):
        # Get the bounding box of the subplot in figure coordinates
        bbox = ax.get_window_extent().transformed(fig.transFigure.inverted())
        y_pos = bbox.y0 + bbox.height / 2  # Vertical center of the subplot
    
        fig.text(label_x_pos, y_pos, row_label_list[i], fontsize= p1m * 70, 
                 horizontalalignment='center', verticalalignment='center', 
                 bbox=dict(facecolor='lightgrey', edgecolor='none', boxstyle='round,pad=0.5'),
                 transform=fig.transFigure)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Save and show the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"flowcell_overview.{plot_format}"))
    plt.close()
    
def plot_multi_flowcell(multi_data, output_dir, plot_format, p1m, q):
    # Ensure the DataFrame has a 'flowcell' column
    if 'flowcell' not in multi_data.columns:
        raise ValueError("DataFrame must contain a 'flowcell' column.")

    # Set mux intervals (modify as needed)
    mux_intervals = np.arange(0, multi_data['hour'].max() + 1, 8)

    # Call functions for each plot type
    plot_length_distributions(multi_data, output_dir, plot_format, p1m, q)
    plot_qscore_distributions(multi_data, output_dir, plot_format, p1m, q)
    plot_yield_over_time_multi(multi_data, output_dir, mux_intervals, plot_format, p1m, q)
    plot_sequence_length_over_time_multi(multi_data, output_dir, mux_intervals, plot_format, p1m, q)
    plot_qscore_over_time_multi(multi_data, output_dir, mux_intervals, plot_format, p1m, q)
    plot_yield_by_length_multi(multi_data, output_dir, plot_format, p1m, q)

def plot_length_distributions(multi_data, output_dir, plot_format, p1m, q):
    # Example: Creating a density plot of read lengths for each flowcell
    plt.figure(figsize=(p1m*960/75, p1m*480/75))
    for flowcell in multi_data['flowcell'].unique():
        flowcell_data = multi_data[multi_data['flowcell'] == flowcell]
        sns.kdeplot(flowcell_data['sequence_length_template'], label=flowcell)
    plt.xscale('log')
    plt.xlabel('Read length (bases)', size = p1m * 15)
    plt.ylabel('Density', size = p1m * 15)
    plt.title('Length Distributions Across Flowcells', size = p1m * 25)
    plt.legend(title='Flowcells', fontsize=p1m * 12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/length_distributions.{plot_format}")
    plt.close()

def plot_qscore_distributions(multi_data, output_dir, plot_format, p1m, q):
    """
    Plot Q score distributions for each flowcell.

    Args:
    multi_data (DataFrame): The DataFrame containing the data to plot.
    output_dir (str): Directory to save the output plot.
    q (float): Q score cutoff.
    p1m (float): Scaling factor for plot size.
    plot_format (str): Format of the saved plot.
    """
    plt.figure(figsize=(p1m * 960/75, p1m * 480/75))

    # Ensure 'flowcell' and 'mean_qscore_template' columns are present
    if 'flowcell' not in multi_data.columns or 'mean_qscore_template' not in multi_data.columns:
        raise ValueError("DataFrame must contain 'flowcell' and 'mean_qscore_template' columns.")

    # Iterate over each flowcell and plot Q score distributions
    for flowcell in multi_data['flowcell'].unique():
        flowcell_data = multi_data[multi_data['flowcell'] == flowcell]
        sns.kdeplot(flowcell_data['mean_qscore_template'], label=flowcell)

    plt.xlabel('Mean Q Score of Read', fontsize=p1m * 15)
    plt.ylabel('Density', fontsize=p1m * 15)
    plt.title('Q Score Distributions Across Flowcells', fontsize=p1m * 25)
    plt.legend(title='Flowcell', fontsize=p1m * 12)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"qscore_distributions.{plot_format}"))
    plt.close()

def plot_yield_over_time_multi(multi_data, output_dir, mux_intervals, plot_format, p1m, q):
    plt.figure(figsize=(p1m*960/75, p1m*480/75))
    for flowcell in multi_data['flowcell'].unique():
        df_flowcell = multi_data[multi_data['flowcell'] == flowcell]
        sns.lineplot(data=df_flowcell, x='hour', y='cumulative.bases.time', label=flowcell)

    # Add vertical lines for mux intervals
    for interval in mux_intervals:
        plt.axvline(x=interval, color='red', linestyle='dashed', alpha=0.5)

    plt.xlabel('Hours Into Run', size=p1m*15)
    plt.ylabel('Total Yield in Gigabases (GB)', size=p1m*15)
    plt.title('Yield Over Time - Multiple Flowcells', size=p1m*25)
    plt.legend(title='Flowcell', fontsize=p1m*12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"yield_over_time_multi.{plot_format}"))
    plt.close()

def plot_yield_by_length_multi(multi_data, output_dir, plot_format, p1m, q):
    plt.figure(figsize=(p1m*960/75, p1m*480/75))
    for flowcell in multi_data['flowcell'].unique():
        df_flowcell = multi_data[multi_data['flowcell'] == flowcell]
        sns.lineplot(data=df_flowcell, x='sequence_length_template', y=df_flowcell['cumulative.bases']/1e9, label=flowcell)

    plt.xlabel('Minimum Read Length (bases)', size=p1m*15)
    plt.ylabel('Total Yield in Gigabases (GB)', size=p1m*15)
    plt.title('Yield by Length - Multiple Flowcells', size=p1m*25)
    plt.legend(title='Flowcell', fontsize=p1m*12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"yield_by_length_multi.{plot_format}"))
    plt.close()

def plot_sequence_length_over_time_multi(multi_data, output_dir, mux_intervals, plot_format, p1m, q):
    plt.figure(figsize=(p1m*960/75, p1m*480/75))
    for flowcell in multi_data['flowcell'].unique():
        df_flowcell = multi_data[multi_data['flowcell'] == flowcell]
        sns.lineplot(data=df_flowcell, x='hour', y='sequence_length_template', label=flowcell)

    # Add vertical lines for mux intervals
    for interval in mux_intervals:
        plt.axvline(x=interval, color='red', linestyle='dashed', alpha=0.5)

    plt.xlabel('Hours Into Run', size=p1m*15)
    plt.ylabel('Sequence Length (bases)', size=p1m*15)
    plt.title('Sequence Length Over Time - Multiple Flowcells', size=p1m*25)
    plt.legend(title='Flowcell', fontsize=p1m*12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"length_over_time_multi.{plot_format}"))
    plt.close()

def plot_qscore_over_time_multi(multi_data, output_dir, mux_intervals, plot_format, p1m, q):
    plt.figure(figsize=(p1m*960/75, p1m*480/75))
    for flowcell in multi_data['flowcell'].unique():
        df_flowcell = multi_data[multi_data['flowcell'] == flowcell]
        sns.lineplot(data=df_flowcell, x='hour', y='mean_qscore_template', label=flowcell)

    # Add vertical lines for mux intervals
    for interval in mux_intervals:
        plt.axvline(x=interval, color='red', linestyle='dashed', alpha=0.5)

    plt.xlabel('Hours Into Run', size=p1m*15)
    plt.ylabel('Mean Q Score', size=p1m*15)
    plt.title('Q Score Over Time - Multiple Flowcells', size=p1m*25)
    plt.legend(title='Flowcell', fontsize=p1m*12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"qscore_over_time_multi.{plot_format}"))
    plt.close()
