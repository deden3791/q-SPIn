# ====== imports ======
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import find_slope
from data_analysis_helpers import apply_axis_style, find_main_branch

# ====== files ======
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

# ====== constants ======
freq_min = 4.4 # GHz - lower bound of frequency region of interest
freq_max = 5.2 # GHz - upper bound of frequency region of interest
amplitude, frequencyGHz, current, fixed_slope, c_ts, dip_currents, dip_freqs, f_c, f_dip = find_slope.find_slope(filepaths, freq_min, freq_max, show=False)

# Peak/dip detection knobs inside the window_halfspan GHz window
window_halfspan = 0.1   # GHz
peak_prominence = 0.45  # smaller -> more peaks found
dip_prominence  = 0.1   # smaller -> more dips found
min_peak_distance_pts = 1

# ====== main loop ======

for filepath in filepaths:
    amplitude_dB = 20 * np.log10(amplitude)
    h5_name = os.path.basename(filepath)
    extent  = [current.min(), current.max(), frequencyGHz.min(), frequencyGHz.max()]

    # --- fit main dispersion branch ---
    c_main, mask_main = find_main_branch(dip_freqs, dip_currents, fixed_slope)
    fit_currents  = np.linspace(current.min(), current.max(), 100)
    fit_freqs_main = fixed_slope * fit_currents + c_main

    # --- horizontal lines to analyse (bright modes + dark mode) ---
    horiz_freqs = [fh for fh in list(f_c) + ([f_dip] if f_dip is not None else [])
                   if freq_min < fh < freq_max]

    # ======= FIGURE 1: two-panel heatmap (left clean, right annotated) =======

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    imshow_kwargs = dict(aspect='auto', extent=extent, origin='lower', cmap='ocean', vmin=-68)
    axL.imshow(amplitude_dB.T, **imshow_kwargs)
    imR = axR.imshow(amplitude_dB.T, **imshow_kwargs)

    axL.text(0.02, 0.98, h5_name, transform=axL.transAxes,
             ha='left', va='top', color='white', fontsize=11, fontweight='bold')

    axL.set_xlabel('Current (A)', fontsize=14)
    axR.set_xlabel('Current (A)', fontsize=14)
    axL.set_ylabel(r'$\omega / 2\pi$ (GHz)', fontsize=14)
    axL.set_xlim(extent[0], extent[1])
    axR.set_xlim(extent[0], extent[1])
    axL.set_ylim(extent[2], extent[3])

    apply_axis_style(axL)
    apply_axis_style(axR)

    cbar = fig.colorbar(imR, ax=axR, pad=0.02)
    cbar.set_label(r'$|S_{21}|$ (dB)', labelpad=10, rotation=90, fontsize=14)
    cbar.ax.tick_params(width=1.5, length=0, direction='inout',
                        labelsize=12, right=True, left=True, top=True, bottom=True)
    cbar.outline.set_linewidth(1.5)

    # overlays on right panel
    axR.scatter(dip_currents[mask_main], dip_freqs[mask_main],
                color='white', s=2, alpha=0.6, zorder=10)
    axR.plot(fit_currents, fit_freqs_main,
             color='white', linestyle='--', linewidth=1.5, zorder=11)
    for fh in horiz_freqs:
        axR.axhline(fh, color='0.7', linestyle='--', alpha=0.8, linewidth=1, zorder=9)

    # ======= analyse each horizontal line =======

    print(f"\n{h5_name}")
    print_lines    = {"bright1": None, "bright2": None, "antimode": None}
    spectra_to_plot = []
    I_min, I_max   = current.min(), current.max()

    for fh in horiz_freqs:
        # find current index where dispersion line crosses this frequency
        I_int = (fh - c_main) / fixed_slope
        if not (I_min <= I_int <= I_max):
            continue

        i_cur  = int(np.argmin(np.abs(current - I_int)))
        I_near = float(current[i_cur])
        spec   = amplitude_dB[i_cur, :]

        # frequency window around the horizontal line
        fmask = ((frequencyGHz >= fh - window_halfspan) &
                 (frequencyGHz <= fh + window_halfspan) &
                 (frequencyGHz >= freq_min) &
                 (frequencyGHz <= freq_max))
        if not np.any(fmask):
            continue

        fwin = frequencyGHz[fmask]
        swin = spec[fmask]

        is_peak_line = bool(np.any(np.isclose(fh, f_c, atol=1e-12)))
        is_dip_line  = (f_dip is not None) and bool(np.isclose(fh, f_dip, atol=1e-12))

        # find features (peaks for bright modes, dips for dark mode)
        if is_peak_line:
            feat_idx, _ = find_peaks(swin, prominence=peak_prominence,
                                     distance=min_peak_distance_pts)
            if len(feat_idx) == 0:
                feat_idx = np.array([int(np.argmax(swin))])
        elif is_dip_line:
            feat_idx, _ = find_peaks(-swin, prominence=dip_prominence,
                                     distance=min_peak_distance_pts)
            if len(feat_idx) == 0:
                feat_idx = np.array([int(np.argmin(swin))])
        else:
            feat_idx = np.array([], dtype=int)

        feat_freqs_all = np.sort(fwin[feat_idx].astype(float))

        if len(feat_freqs_all) <= 1:
            feat_freqs_plot = feat_freqs_all
            df_feat = 0.0
        else:
            feat_freqs_plot = np.array([feat_freqs_all[0], feat_freqs_all[-1]])
            df_feat = float(feat_freqs_plot[-1] - feat_freqs_plot[0])  # GHz

        # annotate on right panel
        if len(feat_freqs_plot) > 0:
            axR.scatter(np.full(len(feat_freqs_plot), I_near), feat_freqs_plot,
                        marker='o', s=30, facecolors='none',
                        edgecolors='red', linewidths=1.6, zorder=13)

        # store result for printing and Figure 2
        if is_peak_line:
            mode_name = "bright1" if fh == np.min(f_c) else "bright2"
        else:
            mode_name = "antimode"

        print_lines[mode_name] = (
            f"{mode_name} | fh={fh:.4f} GHz | "
            f"Δf={df_feat*1e3:.2f} MHz | Δf/2={df_feat*5e2:.2f} MHz"
        )

        spectra_to_plot.append({
            "fwin": fwin.copy(), "swin": swin.copy(),
            "feat_freqs_plot": feat_freqs_plot.tolist(),
        })

    for key in ["bright2", "antimode", "bright1"]:
        if print_lines[key] is not None:
            print(print_lines[key])

    # ======= FIGURE 2: spectra at each horizontal line =======

    if spectra_to_plot:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        for item in spectra_to_plot:
            fwin, swin = item["fwin"], item["swin"]
            ax2.plot(fwin, swin, linewidth=1.2)
            for ff in item["feat_freqs_plot"]:
                j = int(np.argmin(np.abs(fwin - ff)))
                ax2.scatter([fwin[j]], [swin[j]], marker='o', s=30,
                            facecolors='none', edgecolors='red', linewidths=1.6)
        ax2.set_xlabel(r'$\omega / 2\pi$ (GHz)', fontsize=14)
        ax2.set_ylabel(r'$|S_{21}|$ (dB)', fontsize=14)
        ax2.tick_params(axis='both', direction='in', right=True, top=True)
        plt.tight_layout()
        plt.show()