# Carmine Silano
# Mar 1, 2025

# post processing module originally intended to process 
# the output of YuE Stage 2 stems
# but can be used with any inst + vox stem.

import soundfile as sf
import numpy as np
import torch
from scipy.signal import butter, lfilter
from upsample import process_input
from stereo_upmix import MonoToStereoUpmixer
from saturate import dynamic_saturator
from analysis import analyze_audio
from pedalboard import Pedalboard, Compressor, Distortion, HighShelfFilter, HighpassFilter, PeakFilter, Gain, Chorus, LadderFilter, Phaser, Convolution, Reverb, Delay, Limiter
from pedalboard.io import AudioFile
from buss_compressor import buss_compressor
from lufs_leveler import level_to_lufs

# Butterworth High-pass and Low-pass filters
def butter_filter(data, cutoff, sr, filter_type, order=4):
    """Applies a Butterworth high-pass or low-pass filter."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered = lfilter(b, a, data).T
    print(f"butter_filter : Filtered data shape: {filtered.shape}")
    return filtered

def mono_clone_to_stereo(mono_waveform):
    """Converts a mono waveform to a stereo waveform """
    print("mono_clone_to_stereo : Input buffer shape:", mono_waveform.shape)
    mono_waveform = np.squeeze(mono_waveform)  # Removes extra dimensions (e.g., (1, samples) → (samples,))
    
    if mono_waveform.ndim != 1:
        raise ValueError(f"Expected a 1D array, but got shape {mono_waveform.shape}")

    r = np.column_stack((mono_waveform, mono_waveform))  # Now (samples, 2)
    print("mono_clone_to_stereo : Output buffer shape:", r.shape)
    return r


def save_wav_from_numpy(filename, waveform, sample_rate):
    """Saves a NumPy array as a WAV file."""
    print(f"save_wav_from_numpy : waveform shape: {waveform.shape}")
    sf.write(filename, waveform, sample_rate)

def stereo_upmix_stems(npInst, npVox, samplerate, output_dir):
    # Apply high-pass and low-pass filters
    hpInst = butter_filter(npInst, cutoff=200, sr=samplerate, filter_type='high')
    lpInst = butter_filter(npInst, cutoff=200, sr=samplerate, filter_type='low')
    hpVox = butter_filter(npVox, cutoff=200, sr=samplerate, filter_type='high')
    lpVox = butter_filter(npVox, cutoff=200, sr=samplerate, filter_type='low')

    
    # Convert lp versions to stereo
    lpInstStereo = mono_clone_to_stereo(lpInst)
    lpVoxStereo = mono_clone_to_stereo(lpVox)

    # Initialize Stereo Upmixer
    stereo_upmixer = MonoToStereoUpmixer(samplerate)

    # Process hp versions through stereo upmixer
    hpInstStereo = stereo_upmixer.process_buffer(hpInst)
    hpVoxStereo = stereo_upmixer.process_buffer(hpVox)

    # Sum lp and hp versions to create the final stereo mix
    sumInst = lpInstStereo + hpInstStereo
    sumVox = lpVoxStereo + hpVoxStereo

    # Save intermediate and final results
    save_wav_from_numpy(output_dir + "/hpInst.wav", hpInstStereo, samplerate)
    save_wav_from_numpy(output_dir + "/lpInst.wav", lpInstStereo, samplerate)
    save_wav_from_numpy(output_dir + "/hpVox.wav", hpVoxStereo, samplerate)
    save_wav_from_numpy(output_dir + "/lpVox.wav", lpVoxStereo, samplerate)
    save_wav_from_numpy(output_dir + "/sum_stereo_inst.wav", sumInst, samplerate)
    save_wav_from_numpy(output_dir + "/sum_stereo_vox.wav", sumVox, samplerate)

    print("Stereo files saved.")
    return sumInst, sumVox

# create some space for a vocal to sit on top of. 
# also give it some harmonics
def process_instrumental(audio, samplerate):
    fxchain = Pedalboard([
        Compressor(threshold_db=-1.0, 
                   ratio=1.5, 
                   attack_ms=5.0, 
                   release_ms=100.0),
        HighpassFilter(cutoff_frequency_hz = 36),
        PeakFilter(cutoff_frequency_hz = 111, 
                   q = 0.8, 
                   gain_db = 2.7),
        PeakFilter(cutoff_frequency_hz = 406, 
                   q = 2.0, 
                   gain_db = -3.5),
        PeakFilter(cutoff_frequency_hz = 1434, 
                   q = 2.43, 
                   gain_db = -1.3),
        PeakFilter(cutoff_frequency_hz = 3007, 
                   q = 1.0, 
                   gain_db = -1.3),
        PeakFilter(cutoff_frequency_hz = 5220, 
                   q = 0.63, 
                   gain_db = -2.5),
        HighShelfFilter(cutoff_frequency_hz = 11000, 
                        gain_db = -4.3, 
                        q = 0.78),
        #Distortion(drive_db=3),
        #Gain(gain_db=-1.0)
        #Limiter(threshold_db=-0.1)
    ])

    #dist_fxed = distort_exciter(audio, samplerate, drive=16, distortion=33, highpass=4800, wet_mix=-6, dry_mix=0)
    chain_fxed = fxchain(audio, samplerate)
    return chain_fxed

#give the vocals some clarity, a little more of a processed feel
def process_vocals(audio, samplerate):
    fxchain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz = 100),
        PeakFilter(cutoff_frequency_hz = 271, 
                   q = 1.21, 
                   gain_db = -2.1),
        PeakFilter(cutoff_frequency_hz = 518, 
                   q = 0.31, 
                   gain_db = -0.6),
        PeakFilter(cutoff_frequency_hz = 949, 
                   q = 0.84, 
                   gain_db = -5.0),
        PeakFilter(cutoff_frequency_hz = 2696, 
                   q = 1.0, 
                   gain_db = -2.7),
        HighShelfFilter(cutoff_frequency_hz = 10334, 
                        gain_db = 1.1, 
                        q = 0.21),
        Delay(delay_seconds=0.06, 
              feedback=0, 
              mix=0.1),
        Reverb(room_size=0.5,
               damping=0.5,
               wet_level=0.15,
               dry_level=1.0,
               width=1.0,
               freeze_mode=0.0),
        #Distortion(drive_db=.2),
        Gain(gain_db=-6.0)
        #Limiter(threshold_db=-0.1)
    ])

    #dist_fxed = distort_exciter(audio, samplerate)
    chain_fxed = fxchain(audio, samplerate)
    #effected = stereo_upmix(chain_fxed, samplerate, 32)
    return chain_fxed

# This baby is just to make sure nothing clips on final export
def brickwall_limit(audio, samplerate):
    fxchain = Pedalboard([Limiter(threshold_db=-0.2)])
    chain_fxed = fxchain(audio, samplerate)
    return chain_fxed

def post_process_stems(inst_path, 
                       vocal_path, 
                       output_dir, 
                       final_output_path,
                       ddim_steps=50, 
                       guidance_scale=3.5, 
                       model_name="basic", 
                       device="auto", 
                       seed=45):
    
    # Process input upsampling
    upInstWaveform, samplerate = process_input(inst_path, output_dir + "/upsampled_inst.wav",
                                    ddim_steps, guidance_scale, model_name, device, seed)
    
    upVoxWaveform, _ = process_input(vocal_path, output_dir + "/upsampled_vox.wav", 
                                    ddim_steps, guidance_scale, model_name, device, seed)

    # Convert PyTorch tensors to NumPy arrays
    npInst = upInstWaveform.detach().cpu().numpy()
    npVox = upVoxWaveform.detach().cpu().numpy()

    sumInst, sumVox = stereo_upmix_stems(npInst, npVox, samplerate, output_dir)
    satInst = dynamic_saturator(sumInst, 80)
    sat_out = (satInst + sumVox) / 2 #saturated inst + vocal
    save_wav_from_numpy(output_dir + "/sat_sum.wav", sat_out, samplerate)

    inst_analysis = analyze_audio(output_dir + "/sum_stereo_inst.wav")
    vox_analysis = analyze_audio(output_dir + "/sum_stereo_vox.wav")

    #TODO look at analysis and use it to drive rules and changes to the signal

    pbInst = process_instrumental(sumInst, samplerate)
    pbVox = process_vocals(sumVox, samplerate)

    pbSum = (pbInst + pbVox) / 2
    save_wav_from_numpy(output_dir + "/pb_sum.wav", pbSum, samplerate)

    bcAudio = buss_compressor(samplerate, 
                            pbSum, 
                            threshold_db=-8.0, 
                            ratio=4.0, 
                            attack_us=20000.0, 
                            release_ms=250.0, 
                            mix_percent=100.0)
    
    save_wav_from_numpy(output_dir + "/buss_compressed.wav", bcAudio, samplerate)

    lufsAudio = level_to_lufs(bcAudio, samplerate, target_lufs=-11)
    brickwallAudio = brickwall_limit(lufsAudio, samplerate)

    # final output area should be up a dir (/output/)
    save_wav_from_numpy(final_output_path, brickwallAudio, samplerate)




##just for testing
def main():
    """Execute when the script is run directly TESTING ONLY."""
    print("Running the script directly FOR TESTING!")
    post_process_stems("../output/post/recons_inst.wav", #file for testing
                       "../output/post/recons_vox.wav", #file for testing
                       "../output/post") #dir for testing output

if __name__ == "__main__":
    main()
