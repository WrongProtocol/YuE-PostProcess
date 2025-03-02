#!/usr/bin/env python3

#upsample.py pre-processing / chunking code 
# from Tobi on discord from server "twoshot.app"
# updated by Carmine Silano

import os
import logging
import argparse

import torch
import torch.nn.functional as F
import torchaudio

# If you have audiosr as a separate module, keep these imports.
# Otherwise, adapt as needed.
from audiosr import super_resolution, build_model, save_wave

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def cross_fade(chunk1: torch.Tensor, chunk2: torch.Tensor, overlap_samples: int) -> torch.Tensor:
    fade_in = torch.linspace(0, 1, steps=overlap_samples, device=chunk1.device)
    fade_out = 1.0 - fade_in
    return (chunk1 * fade_out) + (chunk2 * fade_in)



def transform_audio_chunked(
    audio: torch.Tensor,
    transform_fn: callable,
    sr: int,
    chunk_length: float,
    overlap: float = 0.1,
    output_sr: int = None
) -> torch.Tensor:
    """
    Processes `audio` in overlapping chunks using `transform_fn`.

    Args:
        audio: shape (..., T_in), e.g. (1, num_samples)
        transform_fn: function that takes a chunk + sr and returns upsampled audio
        sr: sample rate of the *input* audio
        chunk_length: length in seconds of each chunk in *input* domain
        overlap: fraction of chunk_length to cross-fade between consecutive chunks
        output_sr: if not None, we pad the *transformed* final chunk to the expected size
                   in the output domain (chunk_length * output_sr).

    Returns:
        Concatenated audio of all transformed chunks in the *output* domain.
        Does NOT clamp to original duration. That can be done outside.
    """

    num_samples_in = audio.shape[-1]  # at input sr
    chunk_samples_in = int(chunk_length * sr)
    overlap_samples_in = int(chunk_samples_in * overlap)

    processed_chunks = []
    step_size_in = max(chunk_samples_in - overlap_samples_in, 1)
    total_chunks = max(1, (num_samples_in + step_size_in - 1) // step_size_in)

    logger.info(
        f"Processing audio in {total_chunks} chunk(s) of {chunk_length:.2f}s "
        f"with {overlap * 100:.1f}% overlap."
    )

    for idx, start in enumerate(range(0, num_samples_in, step_size_in)):
        # Number of input samples to cross-fade
        overlap_in = 0 if start == 0 else overlap_samples_in
        overlapping = (start != 0)

        end = start + chunk_samples_in
        chunk_in = audio[..., start:end]

        is_final_chunk = (idx == total_chunks - 1)

        # ------------------------------------------------
        # 1) Transform chunk: (1, T_in) -> (1, T_out)
        # ------------------------------------------------
        # transform_fn should handle the SR up-conversion
        transformed_chunk = transform_fn(chunk_in, sr)
        if not isinstance(transformed_chunk, torch.Tensor):
            transformed_chunk = torch.tensor(transformed_chunk)

        # ------------------------------------------------
        # 2) Optional "final chunk" padding in output domain
        # ------------------------------------------------
        if output_sr is not None and is_final_chunk:
            # The expected size of a full chunk at output_sr:
            expected_out = int(chunk_length * output_sr)
            actual_out = transformed_chunk.shape[-1]

            if actual_out < expected_out:
                pad_amount = expected_out - actual_out
                logger.info(
                    f"Padding final chunk from {actual_out} to {expected_out} samples "
                    f"(chunk_length={chunk_length}, output_sr={output_sr})."
                )
                transformed_chunk = F.pad(transformed_chunk, (0, pad_amount))

        # ------------------------------------------------
        # 3) Overlap/cross-fade + append
        # ------------------------------------------------
        if start == 0:
            # First chunk, just add
            processed_chunks.append(transformed_chunk)
        else:
            # Cross-fade with previous chunk
            last_chunk = processed_chunks[-1]

            # Because the *output* chunk length may differ from input chunk length,
            # we measure overlap in the *output* domain. That means your transform_fn
            # might produce fewer/more samples. You can either fix it to a known ratio
            # or do real-time alignment. For simplicity, assume overlap_in in input
            # corresponds 1:1 with the output domain—i.e. same # of samples to fade.
            # If that is not correct, you need a better mapping from overlap_in to overlap_out!
            #
            # For demonstration, we'll say overlap_out = overlap_in * (output_sr / sr)
            # and then clamp if it exceeds chunk lengths.
            overlap_out = int(overlap_in * (output_sr / sr)) if output_sr else overlap_in
            overlap_out = min(overlap_out, last_chunk.shape[-1], transformed_chunk.shape[-1])

            if overlap_out <= 0:
                # No overlap
                processed_chunks.append(transformed_chunk)
            else:
                last_left = last_chunk[..., :-overlap_out]
                last_right = last_chunk[..., -overlap_out:]

                new_left = transformed_chunk[..., :overlap_out]
                new_right = transformed_chunk[..., overlap_out:]

                # Replace last chunk minus its overlap
                processed_chunks[-1] = last_left

                # cross-fade region
                overlap_region = cross_fade(last_right, new_left, overlap_out)
                processed_chunks.append(overlap_region)

                # remainder of new chunk
                processed_chunks.append(new_right)

    logger.info("Concatenating processed chunks.")
    processed_audio = torch.cat(processed_chunks, dim=-1)

    return processed_audio


def transform_audio_per_channel(audio: torch.Tensor, transform_fn: callable) -> torch.Tensor:
    num_channels = audio.shape[0]
    processed_channels = []

    for ch_idx in range(num_channels):
        logger.info(f"Processing channel {ch_idx + 1}/{num_channels} ...")
        mono_channel = audio[ch_idx:ch_idx + 1, :]
        transformed = transform_fn(mono_channel)
        processed_channels.append(transformed)

    return torch.cat(processed_channels, dim=0)


def process_input(input_file, 
                  output_file, 
                  ddim_steps=50, 
                  guidance_scale=3.5, 
                  model_name="basic", 
                  device="auto", 
                  seed=45):
    
    # Build model
    logger.info("Starting AudioSR processing...")
    audiosr_model = build_model(model_name=model_name, device=device)
    logger.info(f"Model '{model_name}' loaded on device '{audiosr_model.device}'.")

    OUTPUT_SAMPLE_RATE = 48000

    # (A) Basic SR transform that will be chunked
    def sr_transform(chunk: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Applies super-resolution on a chunk of shape (1, T) and returns (1, T').

        NOTE: If your super_resolution() returns an extra batch dimension (1, 1, T),
        we squeeze it out below.
        """
        torchaudio.save("../output/post/intermediate_chunk.wav", chunk, sample_rate)
        result = super_resolution(
            audiosr_model,
            "../output/post/intermediate_chunk.wav",
            seed=seed,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps
        )

        return result.squeeze(0)

    # (B) Wrap sr_transform with chunking logic
    def chunked_sr_transform(mono_audio: torch.Tensor) -> torch.Tensor:
        
        return transform_audio_chunked(
            audio=mono_audio,
            transform_fn=sr_transform,
            sr=sr,
            chunk_length=10.24,
            output_sr=OUTPUT_SAMPLE_RATE
        )

    logger.info(f"Loading input audio: {input_file}")
    audio, sr = torchaudio.load(input_file)
    logger.info(f"Loaded audio with {audio.shape[0]} channel(s), sample rate={sr}")

    # Original duration
    input_duration = audio.shape[-1] / sr

    # Apply chunked SR transform per channel
    logger.info("Starting channel-by-channel, chunked super-resolution...")
    waveform = transform_audio_per_channel(audio, chunked_sr_transform)

    # Normalize
    logger.info("Normalizing output waveform...")
    waveform = waveform - torch.mean(waveform)
    max_val = torch.max(torch.abs(waveform))
    waveform = waveform / (max_val + 1e-8)

    # Trim to original duration
    target_length = int(input_duration * OUTPUT_SAMPLE_RATE)
    if waveform.shape[-1] > target_length:
        logger.info("Trimming final waveform to original duration.")
        waveform = waveform[..., :target_length]

    # Save
    logger.info(f"Saving output to {output_file}")
    torchaudio.save(output_file, waveform, OUTPUT_SAMPLE_RATE)
    logger.info("Done!")
    
    return waveform, OUTPUT_SAMPLE_RATE