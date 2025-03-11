# YuE-PostProc: Post-Processing Module

The YuE-PostProc module is the post-processing component of **YuE**, an AI music generator. This module is designed to take instrumental and vocal stems (or any similar audio stems) produced by earlier stages and apply a series of audio processing techniques—such as upsampling, stereo imaging, dynamic saturation, compression, EQ, and loudness leveling—to produce a finished, cohesive, and professional-sounding output.

To run, follow instructions in install_notes.txt.   For subsequent startups (such as on a docker), "conda activate ai" first.
---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Module Components](#module-components)
  - [Upsampling](#upsampling)
  - [Stereo Upmixing](#stereo-upmixing)
  - [Dynamic Saturation](#dynamic-saturation)
  - [Loudness Leveling](#loudness-leveling)
  - [Buss Compression](#buss-compression)
  - [Additional Effects Processing](#additional-effects-processing)
- [Workflow](#workflow)
- [Installation and Dependencies](#installation-and-dependencies)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## Overview

The post-processing pipeline in YuE-PostProc transforms raw upsampled audio stems into a polished final mix. It is tailored for the output of YuE’s Stage 2 but can be applied to any instrumental or vocal stem. The process includes:
- **Upsampling**: Increasing the audio resolution using deep learning–based super-resolution.
- **Stereo Upmixing**: Converting mono signals to stereo using a delay and sum technique.
- **Dynamic Saturation**: Applying a dynamic, sine-shaped saturator to add warmth and character.
- **EQ and Effects**: Using a chain of effects (compression, EQ filters, reverb, etc.) to refine the sound.
- **Dynamic Compression & LUFS Leveling**: Ensuring the final mix is dynamically controlled and meets target loudness standards.

---

## Features

- **Advanced Upsampling**: Utilizes a chunked processing approach with overlap cross-fading to apply super-resolution to audio chunks (see [upsample.py](#upsample)).
- **Stereo Imaging**: Upmixes mono signals to stereo with a delay-based technique accelerated by Numba for efficiency ([stereo_upmix.py](#stereo-upmixing)).
- **Dynamic Saturation**: Applies a dynamic saturator effect based on sine shaping to add harmonic content and character to the sound ([saturate.py](#dynamic-saturation)).
- **Loudness Normalization**: Adjusts the integrated loudness to a target LUFS level using the ITU-R BS.1770 standard and the pyloudnorm package ([lufs_leveler.py](#loudness-leveling)).
- **Buss Compression**: Implements a dynamic range compressor using an accelerated algorithm to glue the mix together ([buss_compressor.py](#buss-compression)).
- **Flexible Effect Chains**: Uses Pedalboard for additional instrument and vocal processing, including EQ, high-pass/low-pass filtering, reverb, delay, and limiting.
  
---

## Module Components

### Upsampling

- **File:** `upsample.py`
- **Functionality:**  
  - Processes input audio using a deep learning–based super-resolution technique.
  - Divides the audio into overlapping chunks, applies a transformation function (super-resolution), and cross-fades between chunks to ensure smooth transitions.
  - Normalizes and trims the output to maintain the original duration.
- **Key Parameters:** `ddim_steps`, `guidance_scale`, `model_name`, `device`, and `seed`.

### Stereo Upmixing

- **File:** `stereo_upmix.py`
- **Functionality:**  
  - Converts mono audio signals to stereo using a delay-based algorithm.
  - Uses a circular delay buffer and a delay/sum technique to create spatial separation between the channels.
  - Accelerated using Numba JIT compilation for real-time processing.

### Dynamic Saturation

- **File:** `saturate.py`
- **Functionality:**  
  - Applies dynamic saturation to stereo audio.
  - Uses sine-based waveshaping to achieve a warm, harmonically rich sound.
  - Allows adjustment of the wet/dry mix via the `mix_pct` parameter.

### Loudness Leveling

- **File:** `lufs_leveler.py`
- **Functionality:**  
  - Measures the integrated loudness of stereo audio using the ITU-R BS.1770 standard.
  - Adjusts gain to achieve a target LUFS level.
  - Prints the measured LUFS and the gain applied for transparency in processing.

### Buss Compression

- **File:** `buss_compressor.py`
- **Functionality:**  
  - Applies dynamic range compression to the audio signal.
  - Implements smoothing for attack and release times, computing gain reduction on a per-sample basis.
  - Uses Numba to accelerate processing, ensuring minimal latency.

### Additional Effects Processing

- **File:** `post_process.py`
- **Functionality:**  
  - Serves as the central hub that ties together upsampling, stereo upmixing, saturation, EQ, and compression.
  - Applies additional processing to both instrumental and vocal stems using Pedalboard:
    - **Instrumental Chain:** Compression, high-pass filtering, multiple peak filters, high-shelf filtering, and limiting.
    - **Vocal Chain:** High-pass filtering, multiple peak filters, reverb, delay, gain adjustment, and limiting.
  - Summates processed stems and applies final buss compression and LUFS leveling.
  - Includes utility functions for file I/O and directory handling.

---

## Workflow

1. **Upsampling:**  
   - Both instrumental and vocal stems are upsampled using the `process_input` function from `upsample.py`.
2. **Stereo Processing:**  
   - The mono signals are filtered and upmixed to stereo via the `stereo_upmix_stems` function in `post_process.py`.
3. **Saturation and Mixing:**  
   - A dynamic saturator is applied to the instrumental stem and then blended with the vocal stem.
4. **Effects Processing:**  
   - Instrumental and vocal signals are processed with dedicated Pedalboard effect chains.
5. **Compression & Leveling:**  
   - The combined mix is compressed using the buss compressor and then loudness-normalized to the target LUFS.
6. **Output:**  
   - Final processed audio is saved in the desired output directory.

---

## Installation and Dependencies

Ensure you have the following dependencies installed:

- **Python Packages:**  
  - `numpy`
  - `soundfile`
  - `pyloudnorm`
  - `torchaudio`
  - `torch`
  - `scipy`
  - `numba`
  - `pedalboard`
  - _Optional:_ `audiosr` (for the upsampling super-resolution model)

You can install the dependencies via pip:

```bash
pip install numpy soundfile pyloudnorm torchaudio torch scipy numba pedalboard
```

If using the upsampling functionality, ensure that the `audiosr` package (or your alternative super-resolution model package) is installed and correctly configured.

---

## Usage

For integration into your workflow, call the `post_process_stems` function with the following parameters:

- `inst_path`: Path to the instrumental stem.
- `vocal_path`: Path to the vocal stem.
- `output_dir`: Directory for intermediate and processed outputs.
- `final_output_path`: Path where the final output will be saved.
- Other parameters such as `ddim_steps`, `guidance_scale`, `model_name`, `device`, and `seed` to control upsampling.

Some of the files can be run directly for testing purposes, there is a __main__ in several of the files, but this was intended to be a quick test during development.
---

## Configuration

Key parameters can be adjusted to tailor the post-processing pipeline to your needs:

- **Upsampling Parameters:**  
  - `ddim_steps`: Controls the number of diffusion steps.
  - `guidance_scale`: Adjusts the model’s adherence to the input prompt.
  - `model_name` and `device`: Specify the super-resolution model and compute device.
- **Saturation Mix:**  
  - Adjust `mix_pct` in `dynamic_saturator` for more or less effect.
- **Loudness Target:**  
  - Change the `target_lufs` in `level_to_lufs` to suit different loudness standards.
- **Compression Settings:**  
  - Modify `threshold_db`, `ratio`, `attack_us`, `release_ms`, and `mix_percent` in the buss compressor to refine dynamic control.
- **Effect Chains:**  
  - The Pedalboard effect chains in `post_process.py` can be further customized to achieve the desired tonal balance.

---

## Troubleshooting

- **File I/O Delays:**  
  - If you encounter issues with files not being written to disk promptly, note that the pipeline includes a brief sleep period to allow the OS to flush writes. Adjust this delay if necessary.
- **Mono/Stereo Mismatches:**  
  - The pipeline assumes specific input shapes (e.g., mono as a 1D array for upmixing). Ensure that input audio is pre-validated or adjust the processing functions accordingly.
- **Dependency Issues:**  
  - Ensure all required packages are installed. For the upsampling model, verify that your deep learning framework (PyTorch) and any custom modules (e.g., `audiosr`) are set up correctly.

---

## Credits for PostProc Module

- **YuE** from the developers listed below this section, and the **YuE community** for early testing on the AI music generation pipeline.
- **Carmine Silano** – Post-processing module and contributor to multiple effects
- **Michael Gruhn** – Inspiration for the stereo upmixing delay and sum technique.
- **Thomas Scott Stillwell** – Basis for the dynamic saturation and buss compressor techniques.
- **Tobi** from **TwoShot.app** - Chunking pre-processor for AudioSR
---

This README provides an overview and detailed explanation of the post-processing capabilities in YuE-PostProc. For further questions or contributions, please refer to the project repository or contact the development team. Enjoy creating polished, professional audio with YuE!

<p align="center">
    <img src="./assets/logo/白底.png" width="400" />
</p>

<p align="center">
    <a href="https://map-yue.github.io/">Demo 🎶</a> &nbsp;|&nbsp; 📑 <a href="">Paper (coming soon)</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot">YuE-s1-7B-anneal-en-cot 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl">YuE-s1-7B-anneal-en-icl 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-cot">YuE-s1-7B-anneal-jp-kr-cot 🤗</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-icl">YuE-s1-7B-anneal-jp-kr-icl 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-cot">YuE-s1-7B-anneal-zh-cot 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-icl">YuE-s1-7B-anneal-zh-icl 🤗</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s2-1B-general">YuE-s2-1B-general 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-upsampler">YuE-upsampler 🤗</a>
</p>

---
Our model's name is **YuE (乐)**. In Chinese, the word means "music" and "happiness." Some of you may find words that start with Yu hard to pronounce. If so, you can just call it "yeah." We wrote a song with our model's name, see [here](assets/logo/yue.mp3).

YuE is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs (lyrics2song). It can generate a complete song, lasting several minutes, that includes both a catchy vocal track and accompaniment track. YuE is capable of modeling diverse genres/languages/vocal techniques. Please visit the [**Demo Page**](https://map-yue.github.io/) for amazing vocal performance.

## News and Updates
* 📌 Join Us on Discord! [<img alt="join discord" src="https://img.shields.io/discord/842440537755353128?color=%237289da&logo=discord"/>](https://discord.gg/ssAyWMnMzu)

* **2025.02.17 🫶** Now YuE supports music continuation and Google Colab! See [YuE-extend by Mozer](https://github.com/Mozer/YuE-extend).
* **2025.02.07 🎉** Get YuE for Windows on [pinokio](https://pinokio.computer).

* **2025.01.30 🔥 Inference Update**: We now support dual-track ICL mode! You can prompt the model with a reference song, and it will generate a new song in a similar style (voice cloning [demo by @abrakjamson](https://x.com/abrakjamson/status/1885932885406093538), music style transfer [demo by @cocktailpeanut](https://x.com/cocktailpeanut/status/1886456240156348674), etc.). Try it out! 🔥🔥🔥 P.S. Be sure to check out the demos first—they're truly impressive. 

* **2025.01.30 🔥 Announcement: A New Era Under Apache 2.0 🔥**: We are thrilled to announce that, in response to overwhelming requests from our community, **YuE** is now officially licensed under the **Apache 2.0** license. We sincerely hope this marks a watershed moment—akin to what Stable Diffusion and LLaMA have achieved in their respective fields—for music generation and creative AI. 🎉🎉🎉

* **2025.01.29 🎉**: We have updated the license description. we **ENCOURAGE** artists and content creators to sample and incorporate outputs generated by our model into their own works, and even monetize them. The only requirement is to credit our name: **YuE by HKUST/M-A-P** (alphabetic order).
* **2025.01.28 🫶**: Thanks to Fahd for creating a tutorial on how to quickly get started with YuE. Here is his [demonstration](https://www.youtube.com/watch?v=RSMNH9GitbA).
* **2025.01.26 🔥**: We have released the **YuE** series.

<br>

---
## TODOs📋
- [ ] Release paper to Arxiv.
- [ ] Example finetune code for enabling BPM control using 🤗 Transformers.
- [ ] Support stemgen mode https://github.com/multimodal-art-projection/YuE/issues/21
- [ ] Support llama.cpp https://github.com/ggerganov/llama.cpp/issues/11467
- [ ] Support transformers tensor parallel. https://github.com/multimodal-art-projection/YuE/issues/7
- [ ] Online serving on huggingface space.
- [ ] Support vLLM and sglang https://github.com/multimodal-art-projection/YuE/issues/66
- [x] Support Colab: [YuE-extend by Mozer](https://github.com/Mozer/YuE-extend)
- [x] Support gradio interface. https://github.com/multimodal-art-projection/YuE/issues/1
- [x] Support dual-track ICL mode.
- [x] Fix "instrumental" naming bug in output files. https://github.com/multimodal-art-projection/YuE/pull/26
- [x] Support seeding https://github.com/multimodal-art-projection/YuE/issues/20
- [x] Allow `--repetition_penalty` to customize repetition penalty. https://github.com/multimodal-art-projection/YuE/issues/45

---

## Hardware and Performance

### **GPU Memory**
YuE requires significant GPU memory for generating long sequences. Below are the recommended configurations:
- **For GPUs with 24GB memory or less**: Run **up to 2 sessions** to avoid out-of-memory (OOM) errors. Thanks to the community, there are [YuE-exllamav2](https://github.com/sgsdxzy/YuE-exllamav2) and [YuEGP](https://github.com/deepbeepmeep/YuEGP) for those with limited GPU resources. While both enhance generation speed and coherence, they may compromise musicality. (P.S. Better prompts & ICL help!)
- **For full song generation** (many sessions, e.g., 4 or more): Use **GPUs with at least 80GB memory**. i.e. H800, A100, or multiple RTX4090s with tensor parallel.

To customize the number of sessions, the interface allows you to specify the desired session count. By default, the model runs **2 sessions** (1 verse + 1 chorus) to avoid OOM issue.

### **Execution Time**
On an **H800 GPU**, generating 30s audio takes **150 seconds**.
On an **RTX 4090 GPU**, generating 30s audio takes approximately **360 seconds**. 

---

## 🪟 Windows Users Quickstart
- For a **one-click installer**, use [Pinokio](https://pinokio.computer).  
- To use **Gradio with Docker**, see: [YuE-for-Windows](https://github.com/sdbds/YuE-for-windows)

## 🐧 Linux/WSL Users Quickstart
For a **quick start**, watch this **video tutorial** by Fahd: [Watch here](https://www.youtube.com/watch?v=RSMNH9GitbA).  
If you're new to **machine learning** or the **command line**, we highly recommend watching this video first.  

To use a **GUI/Gradio** interface, check out:  
- [YuE-exllamav2-UI](https://github.com/WrongProtocol/YuE-exllamav2-UI)
- [YuEGP](https://github.com/deepbeepmeep/YuEGP)
- [YuE-Interface](https://github.com/alisson-anjos/YuE-Interface)  

### 1. Install environment and dependencies
Make sure properly install flash attention 2 to reduce VRAM usage. 
```bash
# We recommend using conda to create a new environment.
conda create -n yue python=3.8 # Python >=3.8 is recommended.
conda activate yue
# install cuda >= 11.8
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
pip install -r <(curl -sSL https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/requirements.txt)

# For saving GPU memory, FlashAttention 2 is mandatory. 
# Without it, long audio may lead to out-of-memory (OOM) errors.
# Be careful about matching the cuda version and flash-attn version
pip install flash-attn --no-build-isolation
```

### 2. Download the infer code and tokenizer
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
# if you don't have root, see https://github.com/git-lfs/git-lfs/issues/4134#issuecomment-1635204943
sudo apt update
sudo apt install git-lfs
git lfs install
git clone https://github.com/multimodal-art-projection/YuE.git

cd YuE/inference/
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

### 3. Run the inference
Now generate music with **YuE** using 🤗 Transformers. Make sure your step [1](#1-install-environment-and-dependencies) and [2](#2-download-the-infer-code-and-tokenizer) are properly set up. 

Note:
- Set `--run_n_segments` to the number of lyric sections if you want to generate a full song. Additionally, you can increase `--stage2_batch_size` based on your available GPU memory.

- You may customize the prompt in `genre.txt` and `lyrics.txt`. See prompt engineering guide [here](#prompt-engineering-guide).

- You can increase `--stage2_batch_size` to speed up the inference, but be careful for OOM.

- LM ckpts will be automatically downloaded from huggingface. 


```bash
# This is the CoT mode.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1
```

We also support music in-context-learning (provide a reference song), there are 2 types: single-track (mix/vocal/instrumental) and dual-track. 

Note: 
- ICL requires a different ckpt, e.g. `m-a-p/YuE-s1-7B-anneal-en-icl`.

- Music ICL generally requires a 30s audio segment. The model will write new songs with similar style of the provided audio, and may improve musicality.

- Dual-track ICL works better in general, requiring both vocal and instrumental tracks.

- For single-track ICL, you can provide a mix, vocal, or instrumental track.

- You can separate the vocal and instrumental tracks using [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) or [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui).

```bash
# This is the dual-track ICL mode.
# To turn on dual-track mode, enable `--use_dual_tracks_prompt`
# and provide `--vocal_track_prompt_path`, `--instrumental_track_prompt_path`, 
# `--prompt_start_time`, and `--prompt_end_time`
# The ref audio is taken from GTZAN test set.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --use_dual_tracks_prompt \
    --vocal_track_prompt_path ../prompt_egs/pop.00001.Vocals.mp3 \
    --instrumental_track_prompt_path ../prompt_egs/pop.00001.Instrumental.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30 
```

```bash
# This is the single-track (mix/vocal/instrumental) ICL mode.
# To turn on single-track ICL, enable `--use_audio_prompt`, 
# and provide `--audio_prompt_path` , `--prompt_start_time`, and `--prompt_end_time`. 
# The ref audio is taken from GTZAN test set.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --use_audio_prompt \
    --audio_prompt_path ../prompt_egs/pop.00001.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30 
```
---
 
## Prompt Engineering Guide
The prompt consists of three parts: genre tags, lyrics, and ref audio.

### Genre Tagging Prompt
1. An example genre tagging prompt can be found [here](prompt_egs/genre.txt).

2. A stable tagging prompt usually consists of five components: genre, instrument, mood, gender, and timbre. All five should be included if possible, separated by space (space delimiter).

3. Although our tags have an open vocabulary, we have provided the top 200 most commonly used [tags](./top_200_tags.json). It is recommended to select tags from this list for more stable results.

3. The order of the tags is flexible. For example, a stable genre tagging prompt might look like: "inspiring female uplifting pop airy vocal electronic bright vocal vocal."

4. Additionally, we have introduced the "Mandarin" and "Cantonese" tags to distinguish between Mandarin and Cantonese, as their lyrics often share similarities.

### Lyrics Prompt
1. An example lyric prompt can be found [here](prompt_egs/lyrics.txt).

2. We support multiple languages, including but not limited to English, Mandarin Chinese, Cantonese, Japanese, and Korean. The default top language distribution during the annealing phase is revealed in [issue 12](https://github.com/multimodal-art-projection/YuE/issues/12#issuecomment-2620845772). A language ID on a specific annealing checkpoint indicates that we have adjusted the mixing ratio to enhance support for that language.

3. The lyrics prompt should be divided into sessions, with structure labels (e.g., [verse], [chorus], [bridge], [outro]) prepended. Each session should be separated by 2 newline character "\n\n".

4. **DONOT** put too many words in a single segment, since each session is around 30s (`--max_new_tokens 3000` by default).

5. We find that [intro] label is less stable, so we recommend starting with [verse] or [chorus].

6. For generating music with no vocal (instrumental only), see [issue 18](https://github.com/multimodal-art-projection/YuE/issues/18).


### Audio Prompt

1. Audio prompt is optional. Providing ref audio for ICL usually increase the good case rate, and result in less diversity since the generated token space is bounded by the ref audio. CoT only (no ref) will result in a more diverse output.

2. We find that dual-track ICL mode gives the best musicality and prompt following. 

3. Use the chorus part of the music as prompt will result in better musicality.

4. Around 30s audio is recommended for ICL.

5. For music continuation, see [YuE-extend by Mozer](https://github.com/Mozer/YuE-extend). Also supports Colab.

---

## License Agreement \& Disclaimer  
- The YuE model (including its weights) is now released under the **Apache License, Version 2.0**. We do not make any profit from this model, and we hope it can be used for the betterment of human creativity.
- **Use & Attribution**: 
    - We encourage artists and content creators to freely incorporate outputs generated by YuE into their own works, including commercial projects. 
    - We encourage attribution to the model’s name (“YuE by HKUST/M-A-P”), especially for public and commercial use. 
- **Originality & Plagiarism**: It is the sole responsibility of creators to ensure that their works, derived from or inspired by YuE outputs, do not plagiarize or unlawfully reproduce existing material. We strongly urge users to perform their own due diligence to avoid copyright infringement or other legal violations.
- **Recommended Labeling**: When uploading works to streaming platforms or sharing them publicly, we **recommend** labeling them with terms such as: “AI-generated”, “YuE-generated", “AI-assisted” or “AI-auxiliated”. This helps maintain transparency about the creative process.
- **Disclaimer of Liability**: 
    - We do not assume any responsibility for the misuse of this model, including (but not limited to) illegal, malicious, or unethical activities. 
    - Users are solely responsible for any content generated using the YuE model and for any consequences arising from its use. 
    - By using this model, you agree that you understand and comply with all applicable laws and regulations regarding your generated content.

---

## Acknowledgements
The project is co-lead by HKUST and M-A-P (alphabetic order). Also thanks moonshot.ai, bytedance, 01.ai, and geely for supporting the project.
A friendly link to HKUST Audio group's [huggingface space](https://huggingface.co/HKUSTAudio). 

We deeply appreciate all the support we received along the way. Long live open-source AI!

---

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@misc{yuan2025yue,
  title={YuE: Open Music Foundation Models for Full-Song Generation},
  author={Ruibin Yuan and Hanfeng Lin and Shawn Guo and Ge Zhang and Jiahao Pan and Yongyi Zang and Haohe Liu and Xingjian Du and Xeron Du and Zhen Ye and Tianyu Zheng and Yinghao Ma and Minghao Liu and Lijun Yu and Zeyue Tian and Ziya Zhou and Liumeng Xue and Xingwei Qu and Yizhi Li and Tianhao Shen and Ziyang Ma and Shangda Wu and Jun Zhan and Chunhui Wang and Yatian Wang and Xiaohuan Zhou and Xiaowei Chi and Xinyue Zhang and Zhenzhu Yang and Yiming Liang and Xiangzhou Wang and Shansong Liu and Lingrui Mei and Peng Li and Yong Chen and Chenghua Lin and Xie Chen and Gus Xia and Zhaoxiang Zhang and Chao Zhang and Wenhu Chen and Xinyu Zhou and Xipeng Qiu and Roger Dannenberg and Jiaheng Liu and Jian Yang and Stephen Huang and Wei Xue and Xu Tan and Yike Guo}, 
  howpublished={\url{https://github.com/multimodal-art-projection/YuE}},
  year={2025},
  note={GitHub repository}
}
```
<br>
