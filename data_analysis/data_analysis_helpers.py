import numpy as np
import h5py
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

def make_colour_figure():
    fig, ax = plt.subplots(figsize=(8, 7))
    
    #ax.set_ylim(6.41, 6.465)
    #ax.set_xlim(-232.3, -227.85)
    #ax.set_xlim(228.85, 231.05)

    plt.rcParams['font.family'] = 'Arial'

    ax.set_xlabel(r"$\omega_0\: / \: 2\pi$ (GHz)", fontsize=20, labelpad=10) #, usetex=True
    ax.set_ylabel(r'$\mathrm{\omega \: / \: 2\pi}$ (GHz)', fontsize=20, labelpad=10) #, usetex=True

    ax.spines['top'].set_linewidth(1.15)
    ax.spines['bottom'].set_linewidth(1.15)
    ax.spines['left'].set_linewidth(1.15)
    ax.spines['right'].set_linewidth(1.15)

    ax.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.15, pad = 5, length=6, top=True, right=True)

    #ax.set_xticks([229, 230, 231])  # Replace with your desired x-axis tick values
    #ax.set_xticks([-229, -230, -231])  # Replace with your desired x-axis tick values
    #ax.set_yticks([6.42, 6.44, 6.46])  # Replace with your desired y-axis tick values

    return fig, ax

def plot_pcolormesh(cur, frequency_GHz, amp, norm):
    
    gamma = 28.025           # Gyromagnetic ratio in GHz/T,
    #cur = (cur*1e3)/gamma                   #H_0 in mT
    
    if norm:
        amp_min = np.min(amp)
        amp_max = np.max(amp)

        amp = amp - amp_min
        scale_factor = 1 / (amp_max - amp_min)
        amp *= scale_factor
        
    fig, ax = make_colour_figure()
    cax = ax.pcolormesh(cur, frequency_GHz, amp.T, shading='auto', cmap='inferno')
    cbar = plt.colorbar(cax, shrink=0.8)
    cbar.set_label(r'S$_{21}$', fontsize=20, labelpad=10, rotation=90)
    #cbar.set_ticks([1.0, 0.5, 0.0])
    cbar.ax.yaxis.set_tick_params(labelsize=18, width=1.5, length=0, direction='inout', right=True, left=True, top=True, bottom=True)
    cbar.ax.set_position([0.82, 0.38, 0.03, 0.5])  # left, bottom, width, height
    cbar.outline.set_linewidth(1.5)
    plt.tight_layout()

def anritsu_SignalProcessing(amp, val):
    if len(amp.shape) == 3:
        depth, rows, cols = amp.shape
        for k in range(depth):
            for i in range(rows):
                for j in range(cols):
                    if amp[k, i, j] < val:
                        continue  # Move to the next element if current one is greater than 0.008

                    # If current value is less than or equal to 0.008, update it and previous values
                    while i > 0 and amp[k, i-1, j] >= val:
                        i -= 1
                    amp[k, i, j] = amp[k, i-1, j] if i > 0 else amp[k, i, j]  # Set current value
    else:
        rows, cols = amp.shape
        for i in range(rows):
            for j in range(cols):
                if amp[i, j] < 1:
                    continue  # Move to the next element if current one is greater than 0.008

                # If current value is less than or equal to 0.008, update it and previous values
                while i > 0 and amp[i-1, j] >= 1:
                    i -= 1
                amp[i, j] = amp[i-1, j] if i > 0 else amp[i, j]  # Set current value

    return amp

def background_removal(amplitude):
    return amplitude - amplitude[0]

def read_h5_2D(filepath):
    """Read amplitude, frequency (GHz), and current from a 2D HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        amplitude = np.array(f['entry']['data0']['amplitude'])
        frequencyGHz = np.array(f['entry']['data0']['frequency']) / 1e9
        phase = np.array(f['entry']['data0']['phase'])

        phase_deg = phase * 360 / (2*np.pi)
        amplitude_dB = 20 * np.log10(np.abs(amplitude))
    return amplitude, amplitude_dB, frequencyGHz, phase_deg

def read_h5_3D(filepath):
    """Read amplitude, frequency (GHz), and current from an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        amplitude = np.array(f['entry']['data0']['amplitude'])
        frequencyGHz = np.array(f['entry']['data0']['frequency']) / 1e9
        current = np.array(f['entry']['data0']['current'])

        amplitude_dB = 20 * np.log10(np.abs(amplitude))
    return amplitude, amplitude_dB, frequencyGHz, current

def cavity_detection(amplitude_dB, frequencyGHz):
    '''Detect cavity resonance and dip frequencies from the reference trace (last current).'''
    ref_trace = amplitude_dB[-1, 30:]
    freq_trim = frequencyGHz[30:]
    peak_idx, _ = find_peaks(ref_trace, prominence=0.5)
    f_c  = freq_trim[peak_idx] if len(peak_idx) > 0 else np.array([])
    f_dip = freq_trim[np.argmin(ref_trace)]
    # peak_indices, _ = find_peaks(amplitude_dB[-1, 30:], prominence=0.5)
    # peak_indices += 30
    # f_c = frequencyGHz[peak_indices]
    # f_c = f_c[(f_c > freq_min) & (f_c < freq_max)]
    return f_c, f_dip

def dip_tracking(amplitude_dB, frequencyGHz, current, freq_min, freq_max, f_c, f_dip, full=False):
    '''Track dip frequencies across currents, excluding cavity and dip regions.'''
    dip_currents, dip_freqs = [], []
    for i in range(len(frequencyGHz)):
        if full:
            row = np.argmin(amplitude_dB[:, i])
            dip_currents.append(current[row])
            dip_freqs.append(frequencyGHz[i])
        else:
            fi = frequencyGHz[i]
            near_cavity = np.any(np.abs(fi - f_c) < 0.025) if len(f_c) > 0 else False
            if fi > freq_min and fi < freq_max and not near_cavity and np.abs(fi - f_dip) > 0.1:
                row = np.argmin(amplitude_dB[:, i])
                dip_currents.append(current[row])
                dip_freqs.append(fi)
    return np.array(dip_currents), np.array(dip_freqs)

# Theil-Sen (a form of simple linear regression): slope = median of all pairwise slopes 
# used to find magnon slope
def theil_sen_slope(x, y, max_pairs=5000):
    """Median of pairwise slopes. Subsample if too many pairs."""
    n = len(x)
    idx = np.arange(n)
    if n * (n - 1) // 2 > max_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=int(np.sqrt(2 * max_pairs)), replace=False)
    xi, yi = x[idx], y[idx]
    i_idx, j_idx = np.triu_indices(len(xi), k=1)
    dx = xi[j_idx] - xi[i_idx]
    dy = yi[j_idx] - yi[i_idx]
    valid = np.abs(dx) > 1e-9
    slopes = dy[valid] / dx[valid]
    return float(np.median(slopes))

def apply_axis_style(ax):
    """Apply consistent tick/spine styling to an axis."""
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', direction='in', width=1.15, pad=5, length=6,
                   labelsize=12, right=True, left=True, top=True, bottom=True)
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis='both', which='minor', direction='in', width=1, pad=5,
                   length=3, right=True, left=True, top=True, bottom=True, color='gray')


def find_main_branch(dip_freqs, dip_currents, fixed_slope):
    """Cluster dip points by intercept histogram; return (c_main, mask_main)."""
    intercepts = dip_freqs - fixed_slope * dip_currents
    hist, bin_edges = np.histogram(intercepts, bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks_hist, _ = find_peaks(hist, prominence=0.1)

    if len(peaks_hist) >= 2:
        sorted_idx = np.argsort(hist[peaks_hist])[::-1]
        c1 = bin_centers[peaks_hist[sorted_idx[0]]]
        c2 = bin_centers[peaks_hist[sorted_idx[1]]]
        mask1 = np.abs(intercepts - c1) < np.abs(intercepts - c2)
        mask_main = mask1 if np.sum(mask1) >= np.sum(~mask1) else ~mask1
    else:
        mask_main = np.ones(len(dip_currents), dtype=bool)

    c_main = float(np.median(intercepts[mask_main]))
    return c_main, mask_main