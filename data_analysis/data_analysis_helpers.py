import numpy as np
import h5py
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator

def read_h5(filepath):
    """Read amplitude, frequency (GHz), and current from an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        amplitude = np.array(f['entry']['data0']['amplitude'])
        frequencyGHz = np.array(f['entry']['data0']['frequency']) / 1e9
        current = np.array(f['entry']['data0']['current'])
    return amplitude, frequencyGHz, current

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