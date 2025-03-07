# Carmine Silano
# Feb 25, 2025
# Implement a dynamic saturator effect that applies dynamic saturation to a stereo audio signal.
# The effect is applied to a stereo audio signal and returns the processed audio.
# The technique is based on the work of Thomas Scott Stillwell.

import numpy as np
import math

def dynamic_saturator(audio, mix_pct=100):
    """
    Apply a stereo dynamic saturation effect to an audio array.

    Parameters:
      audio   : ndarray
                Stereo audio input with shape (n_samples, 2)
      mix_pct : float, optional
                Percentage of the wet (processed) signal to mix in (default is 0)

    Returns:
      ndarray: Processed stereo audio with the same shape as the input.
    """
    # Initialize constants
    pi = math.pi
    halfpi = pi / 2.0

    # Compute mix factors
    mix = mix_pct / 100.0
    mix1 = 1 - mix

    # Clamp the audio samples to the range [-1, 1]
    dry = np.clip(audio, -1, 1)

    # Compute the wet (saturated) signal using sine shaping
    wet = np.sin(dry * halfpi)

    # Mix the dry and wet signals according to the mix percentages
    processed_audio = mix1 * dry + mix * wet

    return processed_audio
