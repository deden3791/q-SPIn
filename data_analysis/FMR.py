import numpy as np
import matplotlib.pyplot as plt
from data_analysis_helpers import background_removal, read_h5_2D, read_h5_3D, anritsu_SignalProcessing, plot_pcolormesh
from pathlib import Path

filepaths_2D = [
    r'measurements\FMR17062026\smiley_microstrip\TGVPGQ_VNA_tracedata.h5',
    ]

filepaths_3D = [
    r'measurements\FMR17062026\CoRu01\TGTY5B_2D_current.h5',
    r'measurements\FMR17062026\CoRu02\TGTYMX_2D_current.h5',
    r'measurements\FMR17062026\CoRu03\TGTXU0_2D_current.h5',
    r'measurements\FMR17062026\CoRu04\TGTYWT_2D_current.h5',
    r'measurements\FMR17062026\CoRu05\TGTZ5Y_2D_current.h5',
    r'measurements\FMR17062026\CoRu06\TGTZHK_2D_current.h5',
    r'measurements\FMR17062026\FeGeLeeds5313Parallel\TGVOXK_2D_current.h5',
    r'measurements\FMR17062026\FeGeLeeds5313Perpedicular\TGVP66_2D_current.h5',
    r'measurements\FMR17062026\GaAs\TGU18C_2D_current.h5',
    r'measurements\FMR17062026\GaAsBigger\TGU1GB_2D_current.h5',
    r'measurements\FMR17062026\MS1\TGU06H_2D_current.h5',
    r'measurements\FMR17062026\MS2\TGU0OQ_2D_current.h5',
    r'measurements\FMR17062026\NiFe2401\TGU0FF_2D_current.h5',
    r'measurements\FMR17062026\NiFe2402\TGU0XM_2D_current.h5',
    r'measurements\FMR17062026\NiFe2403\TGTZX0_2D_current.h5',
    r'measurements\FMR17062026\YIGD7\TGU1QO_2D_current.h5',
    r'measurements\FMR17062026\YIGD8\TGU224_2D_current.h5',
                ]

# for file in filepaths_2D:
#     amplitude, amplitude_dB, frequencyGHz, phase_deg = read_h5_2D(file)
#     # amplitude_dB = background_removal(amplitude_dB)
#     print("Analyzing file:", file)

#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#     extent = [frequencyGHz.min(), frequencyGHz.max(), amplitude_dB.min(), amplitude_dB.max()]

#     axes[0].plot(frequencyGHz, amplitude_dB.T)
#     axes[0].set_xlabel('Frequency (GHz)')
#     axes[0].set_ylabel('Amplitude (dB)')

#     axes[1].plot(frequencyGHz, phase_deg)
#     axes[1].set_xlabel('Frequency (GHz)')
#     axes[1].set_ylabel('Phase (degrees)')

for file in filepaths_3D:
    print("Analysing file:", file)
    sample_name = Path(file).parent.name

    amplitude, amplitude_dB, frequencyGHz, current = read_h5_3D(file)

    current = current[0:len(amplitude)]

    amplitude = anritsu_SignalProcessing(amplitude, 0.002)
    amplitude_dB = 20 * np.log10(amplitude)

    normalised = False

    extent = [current.min(), current.max(),
              frequencyGHz.min(), frequencyGHz.max()]

    plot_pcolormesh(current, frequencyGHz, amplitude_dB, normalised)
    plt.xlabel('Current (A)')
    plt.title(sample_name)
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])

    amplitude_dB = background_removal(amplitude_dB)
    plot_pcolormesh(current, frequencyGHz, amplitude_dB, normalised)
    plt.xlabel('Current (A)')
    plt.title(f'{sample_name} - background removed')
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])

    amplitude_dB= 20*np.log10(np.gradient(np.asarray(amplitude), axis=0))
    plot_pcolormesh(current, frequencyGHz, amplitude_dB, normalised)
    plt.xlabel('Current (A)')
    plt.title(f'{sample_name} - gradient')
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])

    plt.show()