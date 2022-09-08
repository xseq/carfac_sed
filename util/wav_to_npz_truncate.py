# This script processes the data from unbalanced data set
# Do the following to prepare array inputs for the neural network model
# Shuffle wav files
# Split to training and evaluation sets
# Load wav files
# Zero padding
# Extract features
# Saving to npz file


import numpy as np
import librosa
import csv
import os
from datetime import datetime
from playsound import playsound
from scipy.io import wavfile
from preprocessing import get_features
from preprocessing import truncate_signal



# parameters
FS = 44100

os.system('clear')
proj_path = os.path.abspath(os.getcwd())
wav_dev_path = proj_path + '/data/wav_dev/'
wav_eval_path = proj_path + '/data/wav_eval/'
npz_path = proj_path + '/data/npz/'

with open(proj_path + '/csv/categories.csv', newline='') as csvfile:
    categories = np.array(list(csv.reader(csvfile)))

n_categories = len(categories)
x_train = []
x_test = []
y_train = []
y_test = []


# training set
print('Processing Training Set')
for p in range(n_categories):
    label_txt = categories[p, 1]
    print('Processing label: ' + label_txt)
    label_folder = wav_dev_path + label_txt + '/'
    wav_file_list = os.listdir(label_folder)
    for q in range(len(wav_file_list)):
        f_name = label_folder + wav_file_list[q]
        _, data = wavfile.read(f_name)
        data = truncate_signal(data, FS)
        data = data.astype(np.float32, order='C') / 32768.0
        data = np.array(data)
        features = get_features(data, FS)    # shape: (128, 130)
        x_train.append(features)
        y_train.append(int(categories[p, 2]))   # a number that stands for the category


# evaluation set
print('Processing Evaluation Set')
for p in range(n_categories):
    label_txt = categories[p, 1]
    print('Processing label: ' + label_txt)
    label_folder = wav_eval_path + label_txt + '/'
    wav_file_list = os.listdir(label_folder)
    for q in range(len(wav_file_list)):
        f_name = label_folder + wav_file_list[q]
        _, data = wavfile.read(f_name)
        data = truncate_signal(data, FS)
        data = data.astype(np.float32, order='C') / 32768.0
        data = np.array(data)
        features = get_features(data, FS)    # shape: (128, 130)
        x_test.append(features)
        y_test.append(int(categories[p, 2]))   # a number that stands for the category


# saving to npz file
now = datetime.now()
current_time = now.strftime("%Y%m%d_%H%M")
npz_file_name = npz_path + 'data_' + current_time
print(['Saving to npz file: ' + npz_file_name])
np.savez(npz_file_name, 
    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
print('Done.')

