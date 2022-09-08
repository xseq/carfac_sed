# audio signal feature processing for both training and inference

import numpy as np
import librosa
import csv
import os
from playsound import playsound


# zero padding to the intended length
# throw away samples longer than 3 seconds
def zero_padding(data_in, FS_in):
    CLIP_DURATION = 3    # seconds
    data_out = [0] * (CLIP_DURATION * FS_in)
    n_copied_samples = min(len(data_in), len(data_out))
    data_out[:n_copied_samples] = data_in[:n_copied_samples]
    return data_out


# get audio features; using a wrapper of librosa for now
def get_features(data_in, FS_in):
    data_in = zero_padding(data_in, FS_in)
    data_in = np.array(data_in)
    melspectrogram = librosa.feature.melspectrogram(
        y=data_in,
        sr=FS_in,
        n_fft=2048,  # 46 ms
        hop_length=1024,  # 23 ms
        n_mels=128
    )
    return librosa.power_to_db(melspectrogram, ref=np.max)


# truncating signal to a clip somewhat centered on the peak
# this is only for training and evaluation
# zero padding as needed
def truncate_signal(data_in, FS_in):
    TIME_BEFORE_PEAK = 0.5  # seconds
    TIME_AFTER_PEAK = 2.5   # seconds
    samples_before_peak = int(FS_in * TIME_BEFORE_PEAK)
    samples_after_peak = int(FS_in * TIME_AFTER_PEAK)  # including peak
    samples_total = samples_before_peak + samples_after_peak

    data_abs = np.abs(data_in)
    peak_idx = np.argmax(data_abs)
    # zero-padding to avoid over- or under-flow
    data_out = np.concatenate(
        np.zeros(samples_before_peak), 
        data_in,
        np.zeros(samples_after_peak)
    )
    # after zero padding, the peak idx is now the starting point
    data_out = data_out[peak_idx:(peak_idx+samples_total-1)]
    return data_out


