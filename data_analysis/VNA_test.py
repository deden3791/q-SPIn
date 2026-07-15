import numpy as np
import matplotlib.pyplot as plt
from data_analysis_helpers import read_h5_2D

filepaths = [
    r'measurements/VNA_test/TI410B_VNA_tracedata.h5',
    ]

for file in filepaths:
    amplitude, amplitude_dB, frequencyGHz, current = read_h5_2D(file)

    top_gap_amp = amplitude_dB.max()
    top_gap_index = np.argmax(amplitude_dB)
    top_gap_freq = frequencyGHz[top_gap_index]

    bottom_gap_amp = amplitude_dB.min()
    bottom_gap_index = np.argmin(amplitude_dB)
    bottom_gap_freq = frequencyGHz[bottom_gap_index]

    # print(f"Top gap: {top_gap_freq:.4f} GHz, {top_gap_amp:.4f} dB")
    # print(f"Bottom gap: {bottom_gap_freq:.4f} GHz, {bottom_gap_amp:.4f} dB")

    extent = [amplitude_dB.min(), amplitude_dB.max(),
              frequencyGHz.min(), frequencyGHz.max()]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frequencyGHz, amplitude_dB, color='tab:blue', linewidth=1.5)
    ax.set_xlabel('Frequency (GHz)', fontsize=14)
    ax.set_ylabel('Amplitude (dB)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    def add_annotation(x, y, label, x_offset=0, y_offset=0):
        ha = 'center'
        if x_offset > 0:
            ha = 'left'
        elif x_offset < 0:
            ha = 'right'

        ax.annotate(
            label,
            fontsize=12,
            xy=(x, y),
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            ha=ha,
            va='bottom' if y_offset >= 0 else 'top',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.5', alpha=0.9),
            arrowprops=dict(
                arrowstyle='->',
                lw=1.0,
                shrinkA=4,
                shrinkB=4,
                connectionstyle='arc3,rad=0.2',
            ),
        )

    add_annotation(top_gap_freq, top_gap_amp, f'{top_gap_freq:.4f} GHz, {top_gap_amp:.4f} dB', x_offset=12, y_offset=16)
    add_annotation(bottom_gap_freq, bottom_gap_amp, f'{bottom_gap_freq:.4f} GHz, {bottom_gap_amp:.4f} dB', x_offset=-10, y_offset=20)

    plt.tight_layout()
    plt.show()
