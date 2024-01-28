#!/usr/bin/env python
"""
Created on Wed Jan 24 08:53:37 2024

@author: ian.michael.bollinger@gmail.com/researchconsultants@critical.consulting
"""
import argparse, multiprocessing, os, glob
from multiprocessing import Pool

import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib, yaml
matplotlib.use('Agg')

import seaborn as sns
sns.set(style="whitegrid")

from MinIONQC_Plots import plot_both_per_channel, plot_channel_summary_histograms, plot_flowcell_overview, plot_length_histogram
from MinIONQC_Plots import plot_length_vs_q, plot_multi_flowcell, plot_qscore_histogram, plot_qscore_over_time
from MinIONQC_Plots import plot_reads_per_hour, plot_sequence_length_over_time, plot_yield_by_length, plot_yield_over_time

### TOOLS

def reads_gt(d, length):
    return len(d[d['sequence_length_template'] >= length])

def bases_gt(d, length):
    reads = d[d['sequence_length_template'] >= length]
    return reads['sequence_length_template'].sum()

def bin_search(min_idx, max_idx, df, threshold=100000):
    # binary search algorithm, thanks to https://stackoverflow.com/questions/46292438/optimising-a-calculation-on-every-cumulative-subset-of-a-vector-in-r/46303384#46303384
    if min_idx == max_idx:
        return min_idx - 1 if df['sequence_length_template'].iloc[min_idx] < threshold else max_idx - 1

    mid_idx = (min_idx + max_idx) // 2
    n = df['sequence_length_template'].iloc[min_idx + (df['cumulative_bases'] > df['cumulative_bases'].iloc[mid_idx] / 2).idxmax()]
    
    if n >= threshold:
        return bin_search(mid_idx, max_idx, df, threshold)
    else:
        return bin_search(min_idx, mid_idx, df, threshold)

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

def calculate_ultra_reads(data, threshold):
    """
    Calculate the number of 'ultra long' reads, defined as those with N50 > threshold.
    """
    sorted_data = data.sort_values(by='sequence_length_template', ascending=False)
    cumsum = sorted_data['sequence_length_template'].cumsum()
    return (cumsum > cumsum.iloc[-1] / 2).sum()

def add_cols(data, min_q, channel_map):
    """
    Take a sequencing summary file (d), and a minimum Q value you are interested in (min_q)
    Return the same dataframe with additional calculated columns.
    """
    data = data[data['mean_qscore_template'] >= min_q]

    if data.empty:
        logging.error(f"There are no reads with a mean Q score higher than your cutoff of {min_q}. Please choose a lower cutoff and try again.")
        return None

    # Mapping channel if max channel number is 512 or less
    if data['channel'].max() <= 512:
        data = pd.merge(data, channel_map, on="channel", how="left")

    # Sort by start time and calculate cumulative bases over time
    data = data.sort_values(by='start_time')
    data['cumulative.bases.time'] = data['sequence_length_template'].cumsum()

    # Sort by read length for cumulative bases
    data = data.sort_values(by='sequence_length_template', ascending=False)
    data['cumulative.bases'] = data['sequence_length_template'].cumsum()

    # Calculate hour of the run and reads per hour
    data['hour'] = data['start_time'] // 3600
    reads_per_hour = data.groupby('hour').size().reset_index(name='reads_per_hour')
    data = pd.merge(data, reads_per_hour, on='hour', how='left')

    return data

### SUMMARY

def load_summary(filepath, min_q, channel_map):
    """
    Load a sequencing summary and add some info.
    min_q is a list of two values defining two levels of min.q to have.
    """
    # Define columns to read from the file
    columns_to_read = ['channel', 'num_events_template', 'sequence_length_template',
                       'mean_qscore_template', 'sequence_length_2d', 'mean_qscore_2d', 
                       'start_time', 'calibration_strand_genome_template']

    # Read the TSV file with specified columns
    try:
        data = pd.read_csv(filepath, sep='\t', usecols=lambda x: x in columns_to_read)
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return None

    # Check for PromethION data based on max channel number
    is_promethion = data['channel'].max() > 512

    # Remove control sequences from directRNA runs
    if "calibration_strand_genome_template" in data.columns:
        data = data[data["calibration_strand_genome_template"] != "YHR174W"]

    # Process 1D2 or 2D run data
    if "sequence_length_2d" in data.columns and "mean_qscore_2d" in data.columns:
        data['sequence_length_template'] = data['sequence_length_2d']
        data['mean_qscore_template'] = data['mean_qscore_2d']

    # Check if 'sequence_length_2d' and 'mean_qscore_2d' columns exist, and drop them if they do
    columns_to_drop = ['sequence_length_2d', 'mean_qscore_2d']
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns_to_drop, axis=1)

    # Convert columns to numeric
    data['sequence_length_template'] = pd.to_numeric(data['sequence_length_template'], errors='coerce')
    data['mean_qscore_template'] = pd.to_numeric(data['mean_qscore_template'], errors='coerce')
    data['num_events_template'] = pd.to_numeric(data['num_events_template'], errors='coerce')
    data['start_time'] = pd.to_numeric(data['start_time'], errors='coerce')

    # Extracting flowcell name from the file path
    flowcell_name = os.path.basename(os.path.dirname(filepath))

    # Add the 'flowcell' column to the DataFrame
    data['flowcell'] = flowcell_name

    # Calculate events per base if not a PromethION run
    if not is_promethion:
        data['events_per_base'] = data['num_events_template'] / data['sequence_length_template']

    # Processing data with two Q score cutoffs
    all_reads_df = add_cols(data, min_q[0], channel_map)
    q_df = add_cols(data, min_q[1], channel_map)

    # Label data based on Q cutoff
    all_reads_df['Q_cutoff'] = "All reads"
    q_df['Q_cutoff'] = f"Q>={min_q[1]}"

    # Combine data
    combined_data = pd.concat([all_reads_df, q_df], ignore_index=True)
    
    # Return both combined_data and is_promethion
    return combined_data, is_promethion

def channel_summary(data, q_cutoff_label):
    """
    Calculate summaries of what happened in each of the channels of a flowcell.
    Return both detailed summary data and a pivoted table for heatmap plotting.
    """
    # Aggregating data by channel
    grouped_data = data.groupby('channel').agg(
        total_bases=('sequence_length_template', 'sum'),
        total_reads=('sequence_length_template', lambda x: (x >= 0).sum()),
        min_read_length=('sequence_length_template','min'),
        mean_read_length=('sequence_length_template', 'mean'),
        median_read_length=('sequence_length_template', 'median'),
        row=('row', 'mean'),
        col=('col', 'mean')
    ).reset_index()

    # Melt the DataFrame to long format for detailed analysis
    detailed_summary_data = grouped_data.melt(id_vars=['channel', 'row', 'col'])
    detailed_summary_data['Q_cutoff'] = q_cutoff_label

    # Pivot the DataFrame for heatmap plotting
    heatmap_data = grouped_data.pivot(index='row', columns='col', values='total_bases')

    return detailed_summary_data, heatmap_data

def summary_stats(d, Q_cutoff="All reads"):
    """
    Calculate summary stats for a single value of min.q.
    """
    # Filter data based on Q score cutoff
    filtered_data = d[d['Q_cutoff'] == Q_cutoff]

    # Calculate summary statistics
    total_bases = filtered_data['sequence_length_template'].sum()
    total_reads = len(filtered_data)
    N50_length = calculate_N50(filtered_data['sequence_length_template'])
    min_length = filtered_data['sequence_length_template'].min()
    mean_length = filtered_data['sequence_length_template'].mean()
    median_length = filtered_data['sequence_length_template'].median()
    max_length = filtered_data['sequence_length_template'].max()
    mean_q = filtered_data['mean_qscore_template'].mean()
    median_q = filtered_data['mean_qscore_template'].median()

    # Calculate ultra-long reads and bases (max amount of data with N50 > 100KB)
    ultra_reads = calculate_ultra_reads(filtered_data, 100000)
    ultra_bases = sum(filtered_data['sequence_length_template'][:ultra_reads])

    reads_gt_len = {length: len(filtered_data[filtered_data['sequence_length_template'] >= length])
                    for length in [10000, 20000, 50000, 100000, 200000, 500000, 1000000]}
    bases_gt_len = {length: filtered_data[filtered_data['sequence_length_template'] >= length]['sequence_length_template'].sum()
                    for length in [10000, 20000, 50000, 100000, 200000, 500000, 1000000]}

    return {'total_bases': total_bases,
            'total_reads': total_reads,
            'N50_length': N50_length,
            'mean_length': mean_length,
            'min_length': min_length,
            'median_length': median_length,
            'max_length': max_length,
            'mean_q': mean_q,
            'median_q': median_q,
            'ultra_reads': ultra_reads,
            'ultra_bases': ultra_bases,
            'reads_gt_len': reads_gt_len,
            'bases_gt_len': bases_gt_len}

def create_channel_map():
    # Define each segment of the flowcell for R9.5
    p1 = pd.DataFrame({'channel': range(33, 65), 'row': np.repeat(range(1, 5), 8), 'col': np.tile(range(1, 9), 4)})
    p2 = pd.DataFrame({'channel': range(481, 513), 'row': np.repeat(range(5, 9), 8), 'col': np.tile(range(1, 9), 4)})
    p3 = pd.DataFrame({'channel': range(417, 449), 'row': np.repeat(range(9, 13), 8), 'col': np.tile(range(1, 9), 4)})
    p4 = pd.DataFrame({'channel': range(353, 385), 'row': np.repeat(range(13, 17), 8), 'col': np.tile(range(1, 9), 4)})
    p5 = pd.DataFrame({'channel': range(289, 321), 'row': np.repeat(range(17, 21), 8), 'col': np.tile(range(1, 9), 4)})
    p6 = pd.DataFrame({'channel': range(225, 257), 'row': np.repeat(range(21, 25), 8), 'col': np.tile(range(1, 9), 4)})
    p7 = pd.DataFrame({'channel': range(161, 193), 'row': np.repeat(range(25, 29), 8), 'col': np.tile(range(1, 9), 4)})
    p8 = pd.DataFrame({'channel': range(97, 129), 'row': np.repeat(range(29, 33), 8), 'col': np.tile(range(1, 9), 4)})

    q1 = pd.DataFrame({'channel': range(1, 33), 'row': np.repeat(range(1, 5), 8), 'col': np.tile(range(16, 8, -1), 4)})
    q2 = pd.DataFrame({'channel': range(449, 481), 'row': np.repeat(range(5, 9), 8), 'col': np.tile(range(16, 8, -1), 4)})
    q3 = pd.DataFrame({'channel': range(385, 417), 'row': np.repeat(range(9, 13), 8), 'col': np.tile(range(16, 8, -1), 4)})
    q4 = pd.DataFrame({'channel': range(321, 353), 'row': np.repeat(range(13, 17), 8), 'col': np.tile(range(16, 8, -1), 4)})
    q5 = pd.DataFrame({'channel': range(257, 289), 'row': np.repeat(range(17, 21), 8), 'col': np.tile(range(16, 8, -1), 4)})
    q6 = pd.DataFrame({'channel': range(193, 225), 'row': np.repeat(range(21, 25), 8), 'col': np.tile(range(16, 8, -1), 4)})
    q7 = pd.DataFrame({'channel': range(129, 161), 'row': np.repeat(range(25, 29), 8), 'col': np.tile(range(16, 8, -1), 4)})
    q8 = pd.DataFrame({'channel': range(65, 97), 'row': np.repeat(range(29, 33), 8), 'col': np.tile(range(16, 8, -1), 4)})

    # Combine all the mappings into one DataFrame
    channel_map = pd.concat([p1, p2, p3, p4, p5, p6, p7, p8, q1, q2, q3, q4, q5, q6, q7, q8], ignore_index=True)
    return channel_map

### MAIN CALLS

def analyze_single_flowcell(input_file, output_dir, q, plot_format, plot_stat, mux_intervals, p1m):    
    logging.info(f"Loading input file: {input_file}")
    data, is_promethion = load_summary(input_file, [-float('inf'), q], channel_map)
    
    
    if data['channel'].max() <= 512:
        logging.info('**** MinION Flowcell Data Detected ****')
        minion_bool = True
    else:
        logging.info('**** PromethION Flowcell Data Detected ****')
        minion_bool = False
    
    if mux_intervals == 0:
        mux_intervals = int(data['hour'].max())
    muxes = np.arange(0, data['hour'].max() + 1, mux_intervals)
    
    flowcell = os.path.basename(os.path.dirname(input_file))
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    else:
        output_dir = os.path.join(output_dir, flowcell)

    logging.info(f"{flowcell}: Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"{flowcell}: Processing All Reads and Q>={q} cutoff")
    # Generate summary statistics
    all_reads_summary = summary_stats(data, Q_cutoff="All reads")
    q_cutoff_summary = summary_stats(data, Q_cutoff=f"Q>={q}")

    # Calculate channel summary for 'All reads'
    channel_summary_all_reads, heatmap_data_all_reads = channel_summary(data[data['Q_cutoff'] == "All reads"], "All reads")
    
    # Calculate channel summary for Q_cutoff
    channel_summary_q_cutoff, heatmap_data_q_cutoff = channel_summary(data[data['Q_cutoff'] == f"Q>={q}"], f"Q>={q}")
    
    # Use the DataFrame part of the tuple correctly
    bases_q_cutoff = channel_summary_q_cutoff[channel_summary_q_cutoff['variable'] == 'Number of bases per channel']

    logging.info(f"{flowcell}: Generating Plots")

    # Call the flowcell overview plot function (for MinION data only) NEEDS REVIEW!
    if not is_promethion:
        plot_flowcell_overview(data, output_dir, plot_format, p1m, q)

    # Generate plots
    plot_sequence_length_over_time(data, output_dir, muxes, plot_format, p1m, q)
    plot_qscore_over_time(data, output_dir, muxes, plot_format, p1m, q)
    plot_length_histogram(data, output_dir, plot_format, plot_stat, p1m, q)
    plot_qscore_histogram(data, output_dir, plot_format, p1m, q)
    plot_yield_over_time(data, output_dir, muxes, plot_format, p1m, q)
    plot_reads_per_hour(data, output_dir, muxes, plot_format, p1m, q)
    plot_channel_summary_histograms(channel_summary_all_reads, channel_summary_q_cutoff, output_dir, args.format, p1m, q)
    plot_length_vs_q(data, output_dir, args.format, p1m, q)
    plot_both_per_channel(channel_summary_all_reads, channel_summary_q_cutoff, output_dir, args.format, p1m, q)
    plot_yield_by_length(data, output_dir, plot_format, p1m, q)
    
    # Save the summary to a YAML file
    summary = {"input file": input_file,
               "All reads": all_reads_summary,
               f"Q>={q}": q_cutoff_summary,
               "notes": "ultralong reads refers to the largest set of reads with N50>100KB"}
    with open(os.path.join(output_dir, "summary.yaml"), "w") as file:
        yaml.dump(summary, file)

    return data

def analyze_combined_flowcells(multi_data, output_dir, q, plot_format, plot_stat, p1m):
    logging.info("Summarising combined data from all flowcells")

    # Remove unnecessary columns
    drops = ["cumulative.bases", "hour", "reads.per.hour"]
    multi_data = multi_data.drop(columns=drops, errors='ignore')
    
    # Filter and sort data for 'All reads' and 'Q>={q}'
    d1 = multi_data[multi_data['Q_cutoff'] == 'All reads'].sort_values(by='sequence_length_template', ascending=False)
    d2 = multi_data[multi_data['Q_cutoff'] == f'Q>={q}'].sort_values(by='sequence_length_template', ascending=False)

    # Calculate cumulative bases
    d1['cumulative.bases'] = d1['sequence_length_template'].cumsum()
    d2['cumulative.bases'] = d2['sequence_length_template'].cumsum()

    # Combine the data
    combined_data = pd.concat([d1, d2])
    
    # Generate summary statistics
    all_reads_summary = summary_stats(d1, Q_cutoff="All reads")
    q_cutoff_summary = summary_stats(d2, Q_cutoff=f"Q>={q}")

    # Save the summary to a YAML file
    summary = {"All reads": all_reads_summary,
            f"Q>={q}": q_cutoff_summary,
            "notes": "ultralong reads refers to the largest set of reads with N50>100KB"}
    with open(os.path.join(output_dir, "combined_summary.yaml"), "w") as file:
        yaml.dump(summary, file)

    # Generate combined plots
    logging.info("Plotting combined length histogram")
    plot_length_histogram(combined_data, output_dir, plot_format, plot_stat, p1m, q)

    logging.info("Plotting combined mean Q score histogram")
    plot_qscore_histogram(combined_data, output_dir, plot_format, p1m, q)

    logging.info("Plotting combined yield by length")
    plot_yield_by_length(combined_data, output_dir, plot_format, p1m, q)

    return combined_data

### ARG PARSING
parser = argparse.ArgumentParser(description="MinIONQC.py: Python based Quality control for MinION sequencing data")
parser.add_argument("-i", "--input", type=str, required=True,
                    help="Input file or directory (required). Either a full path to a sequence_summary.txt file, or a full path to a directory containing one or more such files. In the latter case, the directory is searched recursively.")
parser.add_argument("-o", "--outputdirectory", type=str, default=None,
                    help="Output directory (optional, default is the same as the input directory).")
parser.add_argument("-q", "--qscore_cutoff", type=float, default=7.0,
                    help="The cutoff value for the mean Q score of a read (default 7).")
parser.add_argument("-p", "--processors", type=int, default=int(round(multiprocessing.cpu_count()*0.8,0)),
                    help="Number of processors to use for the analysis (default 80% of available CPUs).")
parser.add_argument("-s", "--smallfigures", action='store_true', default=False,
                    help="When true, output smaller figures, suitable for publications or presentations.")
parser.add_argument("-c", "--combined_only", action='store_true', default=False, 
                    help="When true, only produce the combined report, not individual reports for each flowcell.")
parser.add_argument("-f", "--format", type=str, default='png',
                    choices=['png', 'pdf', 'ps', 'jpeg', 'tiff', 'bmp'],
                    help="Output format of the plots.")
parser.add_argument("-m", "--muxes", type=float, default=0,
                    help="The value for mux scan used in MinKnow.")
parser.add_argument("-a", "--add_stat", action='store_true', default=True,
                    help="When true, add some basic statistical values on plots.")
args = parser.parse_args()

### SET BASE GLOBALS
channel_map = create_channel_map()    
q = args.qscore_cutoff
p1m = 0.5 if args.smallfigures else 1.0
output_dir = args.outputdirectory if args.outputdirectory else os.path.dirname(args.input)
minion_bool = True

# Main execution flow
if __name__ == "__main__":
    # Check if the input path is a file
    if os.path.isfile(args.input):
        logging.info("**** Analysing the following file ****")
        logging.info(args.input)
        analyze_single_flowcell(args.input, output_dir, args.qscore_cutoff, args.format, args.add_stat, args.muxes, p1m)
        logging.info('**** Analysis complete ****')
        
    # Check if the input path is a directory
    elif os.path.isdir(args.input):
        # Get a list of all sequencing_summary.txt files recursively
        summaries = glob.glob(os.path.join(args.input, '**', 'sequencing_summary.txt'), recursive=True)

        logging.info("**** Analysing the following files ****")
        for summary in summaries:
            logging.info(summary)

        if len(summaries) == 1:
            analyze_single_flowcell(summaries[0], os.path.dirname(summaries[0]), args.qscore_cutoff, args.format, args.add_stat, args.muxes, p1m)
            logging.info('**** Analysis complete ****')

        else:
            results = []
            if not args.combined_only:
                logging.info('**** Multiprocessing Separate Reads ****')
                with Pool(args.processors) as pool:
                    results = pool.starmap(analyze_single_flowcell, [(summary, os.path.dirname(summary), args.qscore_cutoff, args.format, args.add_stat, args.muxes, p1m) for summary in summaries])
            else:
                logging.info('**** Processing Combined Reads ****')
                for summary in summaries:
                    data, _ = load_summary(summary, [-float('inf'), args.qscore_cutoff], channel_map)
                    results.append(data)

            # Combine data from all flowcells
            multi_data = pd.concat(results, ignore_index=True)            
            if multi_data['channel'].max() <= 512:
                logging.info('**** MinION Flowcell Data Detected ****')
                minion_bool = True
            else:
                logging.info('**** PromethION Flowcell Data Detected ****')
                minion_bool = False
            combined_output_dir = args.outputdirectory if args.outputdirectory else os.path.join(args.input, "combinedQC")
            os.makedirs(combined_output_dir, exist_ok=True)
            analyze_combined_flowcells(multi_data, combined_output_dir, args.qscore_cutoff, args.format, args.add_stat, p1m)
            plot_multi_flowcell(multi_data, combined_output_dir, args.format, p1m, q)
            
            # Add citation note
            logging.info('**** Analysis complete ****')
            logging.info('Converted from R to Python by Ian M. Bollinger')
            logging.info('If you use MinIONQC in your published work, please cite:')
            logging.info('R Lanfear, M Schalamun, D Kainer, W Wang, B Schwessinger (2018). MinIONQC: fast and simple quality control for MinION sequencing data, Bioinformatics, bty654')
            logging.info('https://doi.org/10.1093/bioinformatics/bty654')
            
    else:
        logging.warning(f"Couldn't find a sequencing summary file in your input which was: {args.input}")