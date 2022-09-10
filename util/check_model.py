
# Real time sound event detection

import sys
import os

proj_path = os.path.abspath(os.getcwd())
util_path = proj_path + '/util/'
sys.path.insert(0, util_path)

from preprocessing import get_features # pylint: disable=import-error
import pyaudio
import wave
import numpy as np
import csv
import struct
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model  # pylint: disable=no-name-in-module, disable=import-error
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.io import wavfile
import sounddevice as sd
import librosa.display


os.system('clear')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# load model
proj_path = os.path.abspath(os.getcwd())
f_name = proj_path + '/models/model_20220908_2125_0.79.h5'
model = load_model(f_name)
model.summary() 

