import soundfile as sf
import numpy as np
import torch
from scipy.signal import butter, lfilter
from upsample import process_input
from stereo_upmix import MonoToStereoUpmixer
from saturate import dynamic_saturator

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

def post_process_stems(inst_path, 
                       vocal_path, 
                       output_dir, 
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

    finalOut = satInst + sumVox
    save_wav_from_numpy(output_dir + "/output.wav", finalOut, samplerate)

##just for testing
def main():
    """Execute when the script is run directly TESTING ONLY."""
    print("Running the script directly FOR TESTING!")
    post_process_stems("../output/post/recons_inst.mp3", #file
                       "../output/post/recons_vox.mp3", #file
                       "../output/post") #dir

if __name__ == "__main__":
    main()
