# Carmine Silano
# Feb 23, 2025
# Apply dynamic range compression to an audio signal using a buss compressor model.
# Math techniques learned from Thomas Scott Stillwell
# Later rewritten to utilize Numba for JIT acceleration, which made a HUGE difference in performance.

import numpy as np
import math
import numba

@numba.njit
def buss_compressor(samplerate, 
                    audio_array, 
                    threshold_db=-20.0, 
                    ratio=4.0, 
                    attack_us=20000.0, 
                    release_ms=250.0, 
                    mix_percent=100.0):
    """
    Apply compression using dynamic range compression with JIT acceleration.
    
    Parameters:
        samplerate (int): The sample rate of the audio.
        audio_array (np.ndarray): Stereo audio signal (Nx2).
        threshold_db (float): Compression threshold in dB.
        ratio (float): Compression ratio.
        attack_us (float): Attack time in microseconds.
        release_ms (float): Release time in milliseconds.
        mix_percent (float): Wet/dry mix percentage.
    
    Returns:
        np.ndarray: The compressed audio signal.
    """
    # Conversion constants
    log2db = 8.6858896380650365530225783783321  # linear to dB
    db2log = 0.11512925464970228420089957273422  # dB to linear

    # Convert times and mix
    attack_time = attack_us / 1000000.0  # seconds
    release_time = release_ms / 1000.0     # seconds
    mix = mix_percent / 100.0              # fraction

    # Coefficients for smoothing (attack and release)
    atcoef = math.exp(-1.0 / (attack_time * samplerate))
    relcoef = math.exp(-1.0 / (release_time * samplerate))

    # Compression threshold in linear scale
    threshv = math.exp(threshold_db * db2log)

    n_samples = audio_array.shape[0]
    output_audio = np.empty_like(audio_array)

    # Initialize state variables
    rundb = 0.0   # running average level in dB
    runave = 0.0  # running average of the squared level

    for i in range(n_samples):
        # Retrieve current sample for both channels
        ospl0 = audio_array[i, 0]
        ospl1 = audio_array[i, 1]
        aspl0 = abs(ospl0)
        aspl1 = abs(ospl1)

        # Compute signal level (squared maximum of both channels)
        if aspl0 > aspl1:
            maxspl = aspl0 * aspl0
        else:
            maxspl = aspl1 * aspl1

        # Smoothing of the signal level
        runave = maxspl + relcoef * (runave - maxspl)
        # Ensure non-negative for sqrt:
        if runave < 0:
            runave = 0.0
        det = math.sqrt(runave)

        # Compute gain reduction in dB (only if above threshold)
        if det <= 0:
            overdb = 0.0
        else:
            overdb = log2db * math.log(det / threshv)
            if overdb < 0:
                overdb = 0.0

        # Attack/release smoothing for gain reduction
        if overdb > rundb:
            rundb = overdb + atcoef * (rundb - overdb)
        else:
            rundb = overdb + relcoef * (rundb - overdb)

        # Compute gain reduction factor
        gr = -rundb * (ratio - 1.0) / ratio
        grv = math.exp(gr * db2log)

        # Mix dry and compressed signals
        output_audio[i, 0] = ospl0 * grv * mix + ospl0 * (1.0 - mix)
        output_audio[i, 1] = ospl1 * grv * mix + ospl1 * (1.0 - mix)

    return output_audio
