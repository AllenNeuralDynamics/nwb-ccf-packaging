"""top level run script"""

import argparse
from pathlib import Path
import csv
import json
import os
import shutil
from pathlib import Path

from collections import Counter
import numpy as np
from hdmf.common.table import DynamicTable
from pynwb import NWBHDF5IO
import pandas as pd
import matplotlib.pyplot as plt


data_folder = Path("/data/")
scratch_folder = Path("/scratch/")
results_folder = Path("/results/")

def get_unit_locations(units_table):
    unit_locations = []
    for unit_idx in range(len(units_table)):
        print(unit_idx/len(units_table))
        mean_waveforms = units_table['waveform_mean'][unit_idx]
        waveform_maxes = np.min(mean_waveforms, axis=1)
        peak_channel_idx = np.argmax(waveform_maxes)

        detected_electrodes = units_table['electrodes'][unit_idx]
        unit_location = detected_electrodes.iloc[peak_channel_idx].location
        unit_locations.append(unit_location)        

    return unit_locations

def get_region_pairs(electrodes,isi_column):
    location_pairs = {}
    for i in range(len(electrodes)):
        pair = electrodes['location'].iloc[i], isi_column[i]
        if pair not in location_pairs:
            location_pairs[pair] = 1
        else:
            location_pairs[pair] += 1

    sorted_pairs = sorted(location_pairs.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs


def plot_unit_locations(qc_folder,units_table):
    unit_locations = get_unit_locations(units_table)
    string_counts = Counter(unit_locations)
    labels = list(string_counts.keys())
    counts = list(string_counts.values())

    # Adjust figure size based on label count
    plt.figure(figsize=(max(10, len(labels) * 0.4), 6))
    plt.bar(labels, counts)

    # Improve label readability
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Location')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each Location')

    plt.tight_layout()
    plt.savefig(qc_folder / 'location_freqs.png')
    plt.close()


def plot_corrected_location_pairs(electrodes,isi_column,qc_folder):
    electrodes_df = electrodes.to_dataframe()
    sorted_pairs_with_counts = get_region_pairs(electrodes_df,isi_column)
    pairs, counts = zip(*sorted_pairs_with_counts)
    labels = [f"{a}, {b}" for a, b in pairs]

    # Set bar colors: red if locations differ, blue if same
    colors = ['red' if a != b else 'blue' for a, b in pairs]

    plt.figure(figsize=(max(10, len(labels) * 0.5), 6))
    plt.bar(labels, counts, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title('Location Pair Frequencies')
    plt.tight_layout()
    plt.savefig(qc_folder / 'location_pairs.png')
    plt.close()


def output_electrodes(qc_folder, electrodes, isi_corrected_location):
    electrodes_df = electrodes.to_dataframe()
    # should add isi_column to df without modifying original original electrodes
    electrodes_df['isi_corrected_location'] = isi_corrected_location
    electrodes_df.to_csv(qc_folder / 'electrodes.csv')


def run_qc(qc_folder, electrodes, isi_column, units):
    qc_folder.mkdir(parents=True, exist_ok=True)

    assert len(isi_column) == len(electrodes)
    output_electrodes(qc_folder, electrodes, isi_column)

    plot_corrected_location_pairs(electrodes,isi_column,qc_folder)

    plot_unit_locations(nwb.unit,qc_folder)

if __name__ == "__main__":
    run()
