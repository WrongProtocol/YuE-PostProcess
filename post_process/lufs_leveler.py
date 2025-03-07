# Carmine Silano
# Mar 2, 2025

# This script takes a stereo numpy array (shape: (N, 2)) and levels its loudness
# to a target LUFS level using the ITU-R BS.1770 standard via the pyloudnorm package.

import numpy as np
import soundfile as sf
import pyloudnorm as pyln

def level_to_lufs(audio: np.ndarray, 
                  sample_rate: int, 
                  target_lufs: float = -10.0) -> np.ndarray:
    """
    Levels a stereo numpy array to a target LUFS level.

    Parameters:
        audio (np.ndarray): Input stereo audio of shape (N, 2).
        sample_rate (int): Sample rate of the audio signal.
        target_lufs (float): The desired integrated loudness in LUFS (default is -23.0).

    Returns:
        np.ndarray: Loudness-adjusted stereo audio.
    """
    # Create a loudness meter for the given sample rate
    meter = pyln.Meter(sample_rate)
    
    # Measure the integrated loudness of the input audio
    measured_lufs = meter.integrated_loudness(audio)
    print(f"Measured LUFS: {measured_lufs:.2f} dB")
    
    # Calculate the gain (in dB) needed to match the target loudness
    # decivels for amplitude are based on a logarithmic scale defined by the formula
    #           dB = 20 x log10 ( amplitude_out / amplitude_in )
    #   to reverse this and compute the linear multiplier (factor to multiply the amplitude), do
    #           amplitude_out / amplitude_in = 10^(dB/20)
    gain_db = target_lufs - measured_lufs
    gain_linear = 10 ** (gain_db / 20)
    print(f"Applying gain of {gain_db:.2f} dB (linear factor: {gain_linear:.4f})")
    
    # Apply the gain to adjust the loudness
    adjusted_audio = audio * gain_linear

    
        
    return adjusted_audio

# main func JUST FOR TESTING!!
if __name__ == "__main__":
    input_file = "../output/post/buss_compressed.wav"
    output_file = "../output/post/lufs.wav"

    audio, sample_rate = sf.read(input_file)
    
    # Verify that the audio is stereo with shape (N, 2)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("Input audio must be a stereo (N, 2) numpy array.")
    
    target_lufs = -9.0

    adjusted_audio = level_to_lufs(audio, sample_rate, target_lufs)
    
    sf.write(output_file, adjusted_audio, sample_rate)
    print(f"Processed audio saved to {output_file}")
