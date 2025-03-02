import librosa
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import welch
import essentia.standard as es
from madmom.features.onsets import RNNOnsetProcessor
import subprocess

def analyze_audio(file_path):
    results = {}
    
    # Librosa Analysis
    y, sr = librosa.load(file_path)
    results["rms"] = np.mean(librosa.feature.rms(y=y))
    results["spectral_centroid_brightness"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    results["spectral_contrast"] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    # PyDub Analysis
    audio = AudioSegment.from_file(file_path)
    results["peak"] = audio.max_dBFS
    results["rms"] = audio.rms
    
    # SciPy Signal Processing
    sr, data = wavfile.read(file_path)
    freqs, psd = welch(data, sr)
    results["mean_power_spectral_density"] = np.mean(psd)
    
    # Essentia Analysis
    essentia_audio = es.MonoLoader(filename=file_path)()
    results["loudness"] = es.Loudness()(essentia_audio)
    results["spectral_brightness"] = np.mean(es.SpectralCentroidTime()(essentia_audio))

    # Madmom Onset Detection
    onset_processor = RNNOnsetProcessor()
    results["Onset Strength"] = np.mean(onset_processor(file_path))
    
    # FFmpeg Volume Detection
    ffmpeg_cmd = ["ffmpeg", "-i", file_path, "-af", "volumedetect", "-f", "null", "-"]
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        results["FFmpeg Volume Info"] = e.stdout
    
    for key, value in results.items():
        print(f"{key}: {value}")

    return results

#  Main Func JUST FOR TESTING
if __name__ == "__main__":
    file_path = "../output/post/output.wav"  # Change this to your audio file path
    results = analyze_audio(file_path)
    
