import numpy as np
import matplotlib.pyplot as plt
from data_analysis_helpers import read_h5_3D, dip_tracking, cavity_detection, theil_sen_slope

def find_slope(filepaths, freq_min, freq_max, show=False):
    results = {}
    for filepath in filepaths:
        amplitude, amplitude_dB, frequencyGHz, current = read_h5_3D(filepath)
        print("Analyzing file:", filepath)

        f_c, f_dip = cavity_detection(amplitude_dB, frequencyGHz)
        dip_currents, dip_freqs = dip_tracking(
            amplitude_dB, frequencyGHz, current,
            freq_min=freq_min, freq_max=freq_max,
            f_c=f_c, f_dip=f_dip, full=False
        )
        slope_ts = theil_sen_slope(dip_currents, dip_freqs)
        c_ts = np.median(dip_freqs - slope_ts * dip_currents)

        print(f"Theil-Sen slope : {slope_ts:+.4f} GHz/A")

        if show:
            _plot_diagnostic(amplitude_dB, frequencyGHz, current,
                             dip_currents, dip_freqs, slope_ts, c_ts)

        results[filepath] = dict(
            amplitude=amplitude,
            amplitude_dB=amplitude_dB,
            frequencyGHz=frequencyGHz,
            current=current,
            slope_ts=slope_ts,
            c_ts=c_ts,
            dip_currents=dip_currents,
            dip_freqs=dip_freqs,
            f_c=f_c,
            f_dip=f_dip,
        )
    return results


def _plot_diagnostic(amplitude_dB, frequencyGHz, current,
                     dip_currents, dip_freqs, slope_ts, c_ts):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    extent = [current.min(), current.max(), frequencyGHz.min(), frequencyGHz.max()]

    axes[0].imshow(amplitude_dB.T, aspect='auto', extent=extent,
                   origin='lower', cmap='ocean', vmin=-68)
    axes[0].set_xlim(extent[0], extent[1])
    axes[0].set_ylim(extent[2], extent[3])
    axes[0].scatter(dip_currents, dip_freqs, s=2, color='white', alpha=0.5)

    I_plot = np.linspace(current.min(), current.max(), 300)
    axes[0].plot(I_plot, slope_ts * I_plot + c_ts,
                 'r--', linewidth=1.5, label=f'Theil-Sen ({slope_ts:+.3f})')
    axes[0].set_xlabel('Current (A)')
    axes[0].set_ylabel('Frequency (GHz)')
    axes[0].legend(fontsize=9)
    axes[0].set_title('Heatmap + slope fits')

    n = len(dip_currents)
    rng = np.random.default_rng(42)
    sub = rng.choice(n, size=min(n, 200), replace=False)
    xi, yi = dip_currents[sub], dip_freqs[sub]
    ii, jj = np.triu_indices(len(xi), k=1)
    dx = xi[jj] - xi[ii]; dy = yi[jj] - yi[ii]
    pw_slopes = dy[np.abs(dx) > 1e-9] / dx[np.abs(dx) > 1e-9]

    axes[1].hist(pw_slopes, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
    axes[1].axvline(slope_ts, color='red', linestyle='--',
                    label=f'Theil-Sen {slope_ts:+.3f}')
    axes[1].set_xlabel('Pairwise slope (GHz/A)')
    axes[1].set_ylabel('Count')
    axes[1].legend(fontsize=9)
    axes[1].set_title('Pairwise slope distribution')

    print(f"set fixed_slope = {slope_ts:.4f}  in the main script\n")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    find_slope()