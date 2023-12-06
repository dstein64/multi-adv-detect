import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ATTACKS, DEFAULT_WORKSPACE, NUM_MODELS, REPRESENTATIONS_SIZE

matplotlib.use('agg')
# Use Type 42 fonts (TrueType) instead of the default Type 3.
matplotlib.rcParams['pdf.fonttype'] = 42
# The font cache is at ~/.cache/matplotlib. Fonts can be viewed from the files in that directory.
# The directory should be deleted after new fonts are installed, so the cache gets rebuilt.
plt.rcParams['font.family'] = 'TeX Gyre Heros'


def plot_model_wise(outdir, df_control, df_treatment):
    fig, axs = plt.subplots(nrows=len(ATTACKS), ncols=len(ATTACKS), figsize=(12, 6.5))
    min_ = min(
        min(df_control.accuracy_mean - df_control.accuracy_std),
        min(df_treatment.accuracy_mean - df_treatment.accuracy_std))
    max_ = max(
        max(df_control.accuracy_mean + df_control.accuracy_std),
        max(df_treatment.accuracy_mean + df_treatment.accuracy_std))
    xticks = np.arange(df_treatment.num_models.min(), df_treatment.num_models.max() + 1)
    for source_idx, source in enumerate(ATTACKS):
        for target_idx, target in enumerate(ATTACKS):
            df_control_subset = df_control[(df_control.adv_source == source) & (df_control.adv_target == target)]
            df_treatment_subset = df_treatment[(df_treatment.adv_source == source) & (df_treatment.adv_target == target)]
            ax = axs[source_idx, target_idx]
            ax.plot(df_control_subset.num_models, df_control_subset.accuracy_mean, color='C0', marker='D', markersize=4)
            ax.fill_between(
                df_control_subset.num_models,
                df_control_subset.accuracy_mean - df_control_subset.accuracy_std,
                df_control_subset.accuracy_mean + df_control_subset.accuracy_std,
                alpha=0.2,
                facecolor='C0')
            ax.plot(df_treatment_subset.num_models, df_treatment_subset.accuracy_mean, color='C1', marker='s', markersize=4)
            ax.fill_between(
                df_treatment_subset.num_models,
                df_treatment_subset.accuracy_mean - df_treatment_subset.accuracy_std,
                df_treatment_subset.accuracy_mean + df_treatment_subset.accuracy_std,
                alpha=0.2,
                facecolor='C1')
            ax.set_ylim((min_ - 0.01, max_ + 0.01))
            ax.yaxis.grid(which='major', color='#D3D3D3', linestyle='solid')
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))  # y-axis tick spacing
            ax.set_xticks(xticks)
            # Keep x labels only for bottom row
            if source_idx != len(ATTACKS) - 1:
                ax.set_xticklabels([])
            # Keep y labels only for left column
            if target_idx != 0:
                ax.set_yticklabels([])
    # Add x-axis label to bottom row
    for idx in range(len(ATTACKS)):
        axs[len(ATTACKS) - 1, idx].set_xlabel('Number of Detection Models', fontsize=11)
    # Add y-axis label to left column
    for idx in range(len(ATTACKS)):
        axs[idx, 0].set_ylabel('Accuracy', fontsize=11)
    # Add column header to top row
    for idx in range(len(ATTACKS)):
        axs[0, idx].set_title(f'Test Attack:\n{ATTACKS[idx].upper()}')
    # Add row labels to left column
    for idx in range(len(ATTACKS)):
        pad = 40
        ax = axs[idx, 0]
        ax.annotate(
            f'Train Attack:\n{ATTACKS[idx].upper()}',
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords='offset points',
            size='large',
            ha='center',
            va='center'
        )
    fig.tight_layout()
    fig.subplots_adjust(left=0.16, top=0.94)
    fig_path = os.path.join(outdir, 'model_wise_plot.pdf')
    fig.savefig(fig_path)
    plt.close(fig)
    if shutil.which('pdfcrop'):
        with tempfile.TemporaryDirectory() as tmp:
            cropped_path = os.path.join(tmp, 'cropped.pdf')
            subprocess.run(['pdfcrop', fig_path, cropped_path])
            shutil.copyfile(cropped_path, fig_path)


def plot_unit_wise(outdir, df_control, df_treatment):
    fig, axs = plt.subplots(nrows=len(ATTACKS), ncols=len(ATTACKS), figsize=(12, 6.5))
    min_ = min(
        min(df_control.accuracy_mean - df_control.accuracy_std),
        min(df_treatment.accuracy_mean - df_treatment.accuracy_std))
    max_ = max(
        max(df_control.accuracy_mean + df_control.accuracy_std),
        max(df_treatment.accuracy_mean + df_treatment.accuracy_std))
    xticks = np.arange(0, max(NUM_MODELS, REPRESENTATIONS_SIZE) + 1, 200)
    for source_idx, source in enumerate(ATTACKS):
        for target_idx, target in enumerate(ATTACKS):
            df_control_subset = df_control[(df_control.adv_source == source) & (df_control.adv_target == target)]
            df_treatment_subset = df_treatment[(df_treatment.adv_source == source) & (df_treatment.adv_target == target)]
            ax = axs[source_idx, target_idx]
            ax.plot(df_control_subset.num_units, df_control_subset.accuracy_mean, color='C0', marker='D', markersize=4)
            ax.fill_between(
                df_control_subset.num_units,
                df_control_subset.accuracy_mean - df_control_subset.accuracy_std,
                df_control_subset.accuracy_mean + df_control_subset.accuracy_std,
                alpha=0.2,
                facecolor='C0')
            ax.plot(df_treatment_subset.num_units, df_treatment_subset.accuracy_mean, color='C1', marker='s', markersize=4)
            ax.fill_between(
                df_treatment_subset.num_units,
                df_treatment_subset.accuracy_mean - df_treatment_subset.accuracy_std,
                df_treatment_subset.accuracy_mean + df_treatment_subset.accuracy_std,
                alpha=0.2,
                facecolor='C1')
            ax.set_ylim((min_ - 0.01, max_ + 0.01))
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))  # y-axis tick spacing
            ax.set_xticks(xticks)
            ax.yaxis.grid(which='major', color='#D3D3D3', linestyle='solid')
            # Keep x labels only for bottom row
            if source_idx != len(ATTACKS) - 1:
                ax.set_xticklabels([])
            # Keep y labels and ticks for only left row
            if target_idx != 0:
                ax.set_yticklabels([])
    # Add x-axis label to bottom row
    for idx in range(len(ATTACKS)):
        axs[len(ATTACKS) - 1, idx].set_xlabel('Number of Units', fontsize=11)
    # Add y-axis label to left column
    for idx in range(len(ATTACKS)):
        axs[idx, 0].set_ylabel('Accuracy', fontsize=11)
    # Add column header to top row
    for idx in range(len(ATTACKS)):
        axs[0, idx].set_title(f'Test Attack:\n{ATTACKS[idx].upper()}')
    # Add row labels to left column
    for idx in range(len(ATTACKS)):
        pad = 40
        ax = axs[idx, 0]
        ax.annotate(
            f'Train Attack:\n{ATTACKS[idx].upper()}',
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords='offset points',
            size='large',
            ha='center',
            va='center'
        )
    fig.tight_layout()
    fig.subplots_adjust(left=0.16, top=0.94)
    fig_path = os.path.join(outdir, 'unit_wise_plot.pdf')
    fig.savefig(fig_path)
    plt.close(fig)
    if shutil.which('pdfcrop'):
        with tempfile.TemporaryDirectory() as tmp:
            cropped_path = os.path.join(tmp, 'cropped.pdf')
            subprocess.run(['pdfcrop', fig_path, cropped_path])
            shutil.copyfile(cropped_path, fig_path)


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default=DEFAULT_WORKSPACE)
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    outdir = os.path.join(args.workspace, 'plot')
    os.makedirs(outdir, exist_ok=True)
    model_wise_control_df = pd.read_csv(os.path.join(args.workspace, 'detect_model_wise_control', 'aggregated.csv'))
    model_wise_treatment_df = pd.read_csv(os.path.join(args.workspace, 'detect_model_wise_treatment', 'aggregated.csv'))
    plot_model_wise(outdir, model_wise_control_df, model_wise_treatment_df)
    unit_wise_control_df = pd.read_csv(os.path.join(args.workspace, 'detect_unit_wise_control', 'aggregated.csv'))
    unit_wise_treatment_df = pd.read_csv(os.path.join(args.workspace, 'detect_unit_wise_treatment', 'aggregated.csv'))
    plot_unit_wise(outdir, unit_wise_control_df, unit_wise_treatment_df)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
