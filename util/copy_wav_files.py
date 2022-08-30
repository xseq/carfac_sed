# Copying a subset of FSD50k audio data from the dataset folder to a local folder
# Each category has its own folder
# Use conda environment spoken_numbers

import numpy as np
import csv
import os
import shutil


os.system('clear')
proj_path = os.path.abspath(os.getcwd())

# parameters
MAX_WAV_FILE_SIZE = 50000000  # last used: 500k

# change the following to the source path name
wav_src_path = '/media/xuan/XZ/Dataset/FSD50k/FSD50K.eval_audio/'  # change here
wav_dst_path = proj_path + '/data/wav_eval/'  # change here

with open(proj_path + '/csv/categories.csv', newline='') as csvfile:
    categories = np.array(list(csv.reader(csvfile)))
with open(proj_path + '/csv/eval.csv', newline='') as csvfile:   # change here
    dev_list = np.array(list(csv.reader(csvfile)))


# initialization
n_categories = len(categories)
n_dev_data = len(dev_list)
file_count = [0] * n_categories


# creating folders
for p in range(n_categories):
    mk_folder_name = wav_dst_path + categories[p, 1]
    os.mkdir(mk_folder_name)
print('Folders created.')


# moving files
for p in range(1, n_dev_data):
    category_name = dev_list[p, 1]
    src_name = wav_src_path + str(dev_list[p, 0]) + '.wav'
    file_size = os.path.getsize(src_name)
    if file_size < MAX_WAV_FILE_SIZE:
        for q in range(n_categories):
            if category_name == categories[q, 0]:
                file_count[q] += 1
                dst_name = wav_dst_path + categories[q, 1] + \
                    '/' + str(dev_list[p, 0]) + '.wav'
                print('Copying file of category ' + 
                    categories[q, 1] + ': ' + str(dev_list[p, 0]) + '.wav')
                shutil.copy(src_name, dst_name)


# display results
print(' ')
print('Task complete.')
for p in range(n_categories):
    print(categories[p, 1] + ': ' + str(file_count[p]))
print('The number of wav files copied: ' + str(sum(file_count)))



# dev set has 2200 files in total, covering all categories

# eval set results
# Task complete.
# Chink and Clink: 38
# Clapping Hands: 35
# Dropping Coins: 41
# Coughing: 49
# Opening Drawer: 33
# Snapping Fingers: 57
# Jangling Keys: 41
# Knocking Doors: 0    # manually copied 10 files
# Laughing: 50
# Walking: 46
# The number of wav files copied: 390

