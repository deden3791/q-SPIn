import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Define the coordinates for each position (z, x) in mm
coordinates = {'Position 1': (-4.640,1.360),
               'Position 2': (0,1.360),
               'Position 3': (4.640,1.360),
               'Position 4': (-4.640,0),
               'Position 5': (0,0),
               'Position 6': (4.640,0),
               'Position 7': (-4.640,-1.360),
               'Position 8': (0,-1.360),
               'Position 9': (4.640,-1.360)}

filepaths = [
    r"measurements\BigMagnet_MawganRes_0.5mm_21052026\position 1\TFBZPK_2D_current.h5",
    r"measurements\BigMagnet_MawganRes_0.5mm_21052026\position 2\TFC0GE_2D_current.h5",
    r"measurements\BigMagnet_MawganRes_0.5mm_21052026\position 3\TFC0XB_2D_current.h5",
    r"measurements\BigMagnet_MawganRes_0.5mm_21052026\position 4\TFC1GM_2D_current.h5",
    r"measurements\BigMagnet_MawganRes_0.5mm_21052026\position 5\TFC1WW_2D_current.h5",
    r"measurements\BigMagnet_MawganRes_0.5mm_21052026\position 6\TFC2BT_2D_current.h5",
    r"measurements\BigMagnet_MawganRes_0.5mm_21052026\position 7\TFC2S7_2D_current.h5",
    r"measurements\BigMagnet_MawganRes_0.5mm_21052026\position 8\TFC37C_2D_current.h5",
    r"measurements\BigMagnet_MawganRes_0.5mm_21052026\position 9\TFC3Q1_2D_current.h5",
]

freq_min = 4.2   # GHz
freq_max = 4.6   # GHz

position_names = list(coordinates.keys())

def read_h5(filepath):
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
    return f_c, f_dip

def dip_tracking(amplitude_dB, frequencyGHz, current, full=False):
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

def fit_anticrossing(dip_currents, dip_freqs, f_cavity_GHz, slope_ts, c_ts):

    # predict crossing current from magnon line: f_m = f_cavity → I_cross
    I_cross_pred = (f_cavity_GHz - c_ts) / slope_ts

    # window around the predicted crossing (±0.2 A, adjust if needed)
    window = 0.2
    cross_mask = np.abs(dip_currents - I_cross_pred) < window

    I_win = dip_currents[cross_mask]
    f_win = dip_freqs[cross_mask]

    if len(I_win) < 6:
        print(f"  Skipping: only {len(I_win)} points in anticrossing window "
              f"(I_cross_pred={I_cross_pred:.3f} A, f_cavity={f_cavity_GHz:.4f} GHz)")
        return None, None, None, None, None, None

    # now split into upper/lower within the window
    upper_mask = f_win > f_cavity_GHz
    lower_mask = f_win < f_cavity_GHz

    I_up, f_up = I_win[upper_mask], f_win[upper_mask]
    I_lo, f_lo = I_win[lower_mask], f_win[lower_mask]

    if len(I_up) < 3 or len(I_lo) < 3:
        print(f"  Skipping: insufficient points on each branch "
              f"(upper={len(I_up)}, lower={len(I_lo)}) within window. "
              f"Try increasing window size.")
        return None, None, None, None, None, None

    def upper_branch(I, g, slope, intercept):
        f_m = slope * I + intercept
        return (f_cavity_GHz + f_m)/2 + np.sqrt(g**2 + ((f_cavity_GHz - f_m)/2)**2)

    def lower_branch(I, g, slope, intercept):
        f_m = slope * I + intercept
        return (f_cavity_GHz + f_m)/2 - np.sqrt(g**2 + ((f_cavity_GHz - f_m)/2)**2)

    p0 = [0.05, slope_ts, c_ts]
    bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    try:
        popt_up, _ = curve_fit(upper_branch, I_up, f_up, p0=p0, bounds=bounds)
        popt_lo, _ = curve_fit(lower_branch, I_lo, f_lo, p0=p0, bounds=bounds)

        g_up, g_lo = popt_up[0], popt_lo[0]
        g_mean = (g_up + g_lo) / 2

        print(f"  I_cross_pred        : {I_cross_pred:.3f} A")
        print(f"  g from upper branch : {g_up*1000:.2f} MHz")
        print(f"  g from lower branch : {g_lo*1000:.2f} MHz")
        print(f"  g mean              : {g_mean*1000:.2f} MHz")

        slope_mean = (popt_up[1] + popt_lo[1]) / 2
        intercept_mean = (popt_up[2] + popt_lo[2]) / 2
        I_cross = (f_cavity_GHz - intercept_mean) / slope_mean

        f_upper_at_cross = f_cavity_GHz + g_mean
        f_lower_at_cross = f_cavity_GHz - g_mean

        return g_mean, popt_up, popt_lo, I_cross, f_upper_at_cross, f_lower_at_cross

    except RuntimeError as e:
        print(f"  Fit failed: {e}")
        return None, None, None, None, None, None

for filepath in filepaths:
    amplitude, frequencyGHz, current = read_h5(filepath)
    print(r"Analyzing file:", filepath)

    amplitude_dB = 20 * np.log10(amplitude)
    # for 3D measurents (i.e., phase) need to transpose so that rows = currents, cols = frequencies. For 2D (amplitude) it's already correct.
    # amp_slice = amplitude_dB[angle_idx]
    # if amp_slice.shape[0] == len(frequencyGHz):
    #     amp_slice = amp_slice.T          # ensure (n_current, n_freq)

    f_c, f_dip = cavity_detection(amplitude_dB, frequencyGHz)
    dip_currents, dip_freqs = dip_tracking(amplitude_dB, frequencyGHz, current, full=False)
    dip_currents_full, dip_freqs_full = dip_tracking(amplitude_dB, frequencyGHz, current, full=True)
    slope_ts = theil_sen_slope(dip_currents, dip_freqs)
    c_ts = np.median(dip_freqs - slope_ts * dip_currents)

    print(f"Cavity frequencies : {f_c} GHz")
    print(f"Dip frequency : {f_dip:.4f} GHz")
    print(f"Theil-Sen slope : {slope_ts:+.4f} GHz/A")

    # ---- anticrossing fit ----
    f_cavity_GHz = float(f_c[1])
    g, popt_up, popt_lo, I_cross, f_up_cross, f_lo_cross = fit_anticrossing(dip_currents_full, dip_freqs_full,
                                        f_cavity_GHz, slope_ts, c_ts)

    # ---- diagnostic plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: heatmap with both fits overlaid
    extent = [current.min(), current.max(), frequencyGHz.min(), frequencyGHz.max()]
    if g is not None:
        axes[0].scatter([I_cross, I_cross], [f_up_cross, f_lo_cross],
                        s=80, facecolors='none', edgecolors='red',
                        linewidths=1.5, zorder=5,
                        label=f'g = {g*1000:.1f} MHz')
        axes[0].annotate(f'+g', (I_cross, f_up_cross), textcoords='offset points',
                        xytext=(5, 3), color='red', fontsize=8)
        axes[0].annotate(f'-g', (I_cross, f_lo_cross), textcoords='offset points',
                        xytext=(5, -8), color='red', fontsize=8)
    axes[0].imshow(amplitude_dB.T, aspect='auto', extent=extent,
                origin='lower', cmap='ocean', vmin=-68)
    axes[0].set_xlim(extent[0], extent[1])
    axes[0].set_ylim(extent[2], extent[3])
    axes[0].scatter(dip_currents, dip_freqs, s=2, color='white', alpha=0.5)

    I_plot = np.linspace(current.min(), current.max(), 300)
    c_ts   = np.median(dip_freqs - slope_ts * dip_currents)
    axes[0].plot(I_plot, slope_ts   * I_plot + c_ts,
                'r--', linewidth=1.5, label=f'Theil-Sen ({slope_ts:+.3f})')
    axes[0].axhline(f_dip, color='grey', linestyle='--')
    for fc in f_c:
        axes[0].axhline(fc, color='grey', linestyle='--')
    axes[0].set_xlabel('Current (A)'); axes[0].set_ylabel('Frequency (GHz)')
    axes[0].legend(fontsize=9)
    axes[0].set_title('Heatmap + slope fits')

    # Right: histogram of pairwise slopes so you can see if there's one clear peak
    n = len(dip_currents)
    rng = np.random.default_rng(42)
    sub = rng.choice(n, size=min(n, 200), replace=False)
    xi, yi = dip_currents[sub], dip_freqs[sub]
    ii, jj = np.triu_indices(len(xi), k=1)
    dx = xi[jj] - xi[ii]; dy = yi[jj] - yi[ii]
    pw_slopes = dy[np.abs(dx) > 1e-9] / dx[np.abs(dx) > 1e-9]

    axes[1].hist(pw_slopes, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
    axes[1].axvline(slope_ts,   color='red',   linestyle='--', label=f'Theil-Sen {slope_ts:+.3f}')
    axes[1].set_xlabel('Pairwise slope (GHz/A)'); axes[1].set_ylabel('Count')
    axes[1].legend(fontsize=9)
    axes[1].set_title('Pairwise slope distribution')

    # ---- paste this into the main script ----
    print(f"set fixed_slope = {slope_ts:.4f}  in the main script\n")

    plt.tight_layout()
    plt.show()