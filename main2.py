#!/usr/bin/env python3
"""
analyze_flyover.py

Complete script for:
- loading flyover sound pressure data (Pa),
- computing chunked RMS (effective pressure) and OSPL (Q5),
- computing PSD per chunk and OSPL from spectrum (Q7),
- plotting waveform with effective pressures, OSPL comparison, and spectrogram.

Usage: python analyze_flyover.py
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
import pandas as pd
import sys

# -----------------------------
# USER PARAMETERS
# -----------------------------
MAT_FILENAME = "Flyover_No_5.mat"   # change if your filename differs
SOUND_VAR_NAMES = ["sound_pressure", "soundPressure", "p", "pressure", "y"]  # candidates to try
FS = 40000.0           # sampling frequency (Hz)
CHUNK_DURATION = 0.1   # seconds (T)
P_REF = 20e-5          # reference sound pressure in Pa (20 µPa)
WINDOW = "hann"        # window to use for periodogram
DETREND = False        # do not detrend chunks (for raw energy consistency)

# -----------------------------
# helper: load .mat and extract vector
# -----------------------------
def load_sound_from_mat(filename, candidate_names=None):
    m = scipy.io.loadmat(filename)
    # print variable keys for transparency
    keys = [k for k in m.keys() if not k.startswith("__")]
    print("Loaded .mat keys:", keys)
    # try candidate names first
    if candidate_names:
        for name in candidate_names:
            if name in m:
                arr = m[name]
                return np.asarray(arr).squeeze()
    # otherwise try to find a 1D numeric array automatically:
    for k in keys:
        v = m[k]
        v = np.asarray(v)
        if v.ndim == 2:
            # flatten single-column or single-row arrays
            if 1 in v.shape:
                return v.flatten()
        elif v.ndim == 1:
            return v
    raise ValueError("Could not find suitable 1D sound array in .mat file. "
                     "Please check variable names or content.")

# -----------------------------
# compute per-chunk metrics
# -----------------------------
def analyze_chunks(signal_arr, fs, chunk_duration, window=WINDOW, detrend=DETREND, pref=P_REF):
    N = len(signal_arr)
    T = chunk_duration
    Nchunk = int(round(fs * T))
    if Nchunk <= 0:
        raise ValueError("chunk duration too small for given fs")
    n_chunks = N // Nchunk   # discard incomplete last chunk
    if n_chunks == 0:
        raise ValueError("Signal shorter than one chunk. Increase data or decrease chunk_duration.")

    times_center = np.zeros(n_chunks)
    effective_pressures = np.zeros(n_chunks)
    OSPL_time = np.zeros(n_chunks)
    OSPL_freq = np.zeros(n_chunks)
    psd_matrix = None
    freqs = None

    for i in range(n_chunks):
        start = i * Nchunk
        stop = start + Nchunk
        chunk = signal_arr[start:stop]

        # center time for plotting
        times_center[i] = (start + stop) / 2.0 / fs

        # time-domain: RMS (effective pressure)
        # discrete integration: p_rms = sqrt( (1/T) * sum(p[n]^2 * dt) )
        dt = 1.0 / fs
        p_rms = np.sqrt(np.sum(chunk ** 2) * dt / T)
        effective_pressures[i] = p_rms
        OSPL_time[i] = 20.0 * np.log10(p_rms / pref)

        # frequency-domain: PSD via periodogram (density in Pa^2/Hz)
        # use nfft = Nchunk so df = 1/T
        f, Pxx = signal.periodogram(chunk,
                                     fs=fs,
                                     window=window,
                                     nfft=Nchunk,
                                     detrend=(None if not detrend else "linear"),
                                     scaling="density")
        # On first chunk, allocate freq matrix
        if freqs is None:
            freqs = f
            psd_matrix = np.zeros((n_chunks, len(freqs)))
        # store PSD (Pa^2/Hz)
        psd_matrix[i, :] = Pxx

        # convert PSD to power per bin: P_bin = PSD * df
        dfreq = f[1] - f[0] if len(f) > 1 else 1.0 / T
        P_bin = Pxx * dfreq                     # units: Pa^2
        total_power = np.sum(P_bin)             # Pa^2
        p_rms_from_spec = np.sqrt(total_power)  # Pa
        OSPL_freq[i] = 20.0 * np.log10(p_rms_from_spec / pref)

    return {
        "times_center": times_center,
        "effective_pressures": effective_pressures,
        "OSPL_time": OSPL_time,
        "OSPL_freq": OSPL_freq,
        "psd_matrix": psd_matrix,
        "freqs": freqs,
        "Nchunk": Nchunk,
        "n_chunks": n_chunks
    }

# -----------------------------
# plotting helpers
# -----------------------------
def plot_waveform_with_effective(time_axis, signal_arr, times_center, eff_pressures):
    plt.figure(figsize=(12, 4.5))
    plt.plot(time_axis, signal_arr, linewidth=0.6, label="Signal (Pa)")
    plt.plot(times_center, eff_pressures, linewidth=2.0, color="green", label="Effective pressure (RMS per chunk)")
    plt.title("Figure 1: Waveform with Effective Pressure (green)")
    plt.xlabel("Time (s)")
    plt.ylabel("Sound pressure (Pa)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_OSPL_comparison(times_center, OSPL_time, OSPL_freq):
    plt.figure(figsize=(12, 4.5))
    plt.plot(times_center, OSPL_time, color="red", linewidth=1.5, label="Time-domain OSPL (Q5)")
    plt.plot(times_center, OSPL_freq, '--', color="blue", linewidth=1.2, label="Freq-domain OSPL (Q7)")
    plt.title("Figure 2: OSPL Comparison (time-domain vs frequency-domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("OSPL (dB re 20 µPa)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_spectrogram(psd_matrix, freqs, times_center):
    # Convert PSD (Pa^2/Hz) to dB (10*log10 since it's power)
    S_dB = 10.0 * np.log10(psd_matrix + 1e-20)  # protect against log(0)
    plt.figure(figsize=(12, 6))
    extent = [times_center[0], times_center[-1], freqs[0], freqs[-1]]
    plt.imshow(S_dB.T, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(label="PSD (dB re Pa^2/Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Figure 3: Spectrogram (periodogram per chunk)")
    plt.tight_layout()
    plt.show()

# -----------------------------
# main
# -----------------------------
def main():
    # Load .mat
    try:
        sound = load_sound_from_mat(MAT_FILENAME, SOUND_VAR_NAMES)
    except Exception as e:
        print("Error loading .mat:", e)
        sys.exit(1)

    # Make sure it's 1D numeric
    sound = np.asarray(sound, dtype=float).squeeze()
    if sound.ndim != 1:
        print("Loaded sound array is not 1D. Exiting.")
        sys.exit(1)

    N = len(sound)
    duration = N / FS
    print(f"Signal length: {N} samples, Duration: {duration:.6f} s")

    # time axis for waveform
    time_axis = np.arange(N) / FS

    # chunk / sample settings
    Nchunk_expected = int(round(FS * CHUNK_DURATION))
    print(f"Chunk duration T={CHUNK_DURATION} s => samples per chunk Nchunk={Nchunk_expected}")

    # analyze
    results = analyze_chunks(sound, FS, CHUNK_DURATION, window=WINDOW, detrend=DETREND, pref=P_REF)

    # quick sanity prints
    print(f"Processed {results['n_chunks']} full chunks of {results['Nchunk']} samples each")
    print(f"PSD matrix shape: {results['psd_matrix'].shape}, freq length: {len(results['freqs'])}")

    # Plots
    plot_waveform_with_effective(time_axis, sound, results['times_center'], results['effective_pressures'])
    plot_OSPL_comparison(results['times_center'], results['OSPL_time'], results['OSPL_freq'])
    plot_spectrogram(results['psd_matrix'], results['freqs'], results['times_center'])

    # Optional: print mean absolute difference between Q5 and Q7 (should be small)
    diff = results['OSPL_time'] - results['OSPL_freq']
    print(f"OSPL difference (time - freq): mean={np.mean(diff):.4f} dB, std={np.std(diff):.4f} dB")

if __name__ == "__main__":
    main()
