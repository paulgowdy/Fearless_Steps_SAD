import glob
import re
from scipy.io.wavfile import read
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pickle

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

#sample_transcript = transcript_files[0]
#sample_audio = audio_files[0]

def read_transcript_file(transcript_fn):

    '''
    Given a transcript filename
    open the transcript file, read the lines
    return all the intervals, with labels
    '''

    with open(transcript_fn, 'r') as f:

        z = f.readlines()

    intervals = [[x.split()[0], float(x.split()[2]), float(x.split()[3]), float(x.split()[3]) - float(x.split()[2]), x.split()[4]] for x in z]

    return intervals

def read_audio_file_to_spect(audio_fn):

    '''
    open a wav file and produce the spect
    '''

    sample_wav = read(audio_fn)

    fs = sample_wav[0]

    f, t, Sxx = signal.spectrogram(sample_wav[1], fs, mode = 'complex')

    return f,t, np.abs(Sxx)

def intervals_to_labels(time_bins, intervals):

    labels = []
    #fs = 8000

    current_interval = 0
    current_interval_end_time = intervals[0][2]

    if intervals[0][4] == 'NS':

        current_label = -1

    elif intervals[0][4] == 'S':

        current_label = 1

    else:

        print('Interval list is messed up')
        return 'ERROR!'


    for time_bin in time_bins:

        #print(time_bin, current_interval_end_time)

        labels.append(current_label)

        if time_bin > current_interval_end_time:

            current_label *= -1

            current_interval += 1

            if current_interval < len(intervals):
                current_interval_end_time = intervals[current_interval][2]

    return np.clip(np.array(labels), 0, 1)

'''
def spect_chop_labels(labels, spect, window_time_width = 4, freq_start = 10, window_freq_height = 60):

    if len(labels) != spect.shape[1]:

        print("Shapes dont match!")
        print(labels.shape)
        print(spect.shape)
        return "Error!"

    spect_slices = []

    for i in range(len(labels) - window_time_width + 1):

        spect_slice = spect[freq_start : window_freq_height, i : i + window_time_width]

        ss = spect_slice.flatten()

        spect_slices.append(ss)

    return np.array(spect_slices), labels[: - window_time_width + 1]
'''

def spect_chop_labels(labels, spect, window_time_width = 4, freq_start = 0, window_freq_height = 60):

    if len(labels) != spect.shape[1]:

        print("Shapes dont match!")
        print(labels.shape)
        print(spect.shape)
        return "Error!"

    spect_slices = []
    new_labels = []

    for i in range(len(labels) - window_time_width + 1):

        c_label =  labels[i]

        spect_slice = spect[freq_start : window_freq_height, i : i + window_time_width]
        ss = spect_slice
        #ss = spect_slice.flatten()

        spect_slices.append(ss)
        new_labels.append(c_label)

        if c_label == 1:

            for _ in range(3):

                spect_slices.append(ss)
                new_labels.append(c_label)



    return np.array(spect_slices), new_labels

def fn_to_data(transcript_fn, audio_fn):

    intervals = read_transcript_file(transcript_fn)

    f, t, S = read_audio_file_to_spect(audio_fn)

    labelz = intervals_to_labels(t, intervals)

    spect_chops, trunc_labels = spect_chop_labels(labelz, S, window_time_width = 20, window_freq_height = 60)

    # one_hot_encode labels...

    ohe_labels = indices_to_one_hot(trunc_labels, 2)

    return spect_chops, ohe_labels

def create_all_data():

    transcript_files = sorted_nicely(glob.glob('Data/Transcripts/SAD/Dev/*.txt'))

    audio_files = sorted_nicely(glob.glob('Data/Audio/Tracks/Dev/*.wav'))

    x_train = []
    y_train = []

    x_val = []
    y_val = []

    for f in range(len(transcript_files)):

        print(f)

        trans_fn = transcript_files[f]
        audio_fn = audio_files[f]

        spects, labels = fn_to_data(trans_fn, audio_fn)

        if f % 10 == 0:

            x_val.extend(list(spects))
            y_val.extend(list(labels))

        else:

            x_train.extend(list(spects))
            y_train.extend(list(labels))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

    with open('pg_data/SAD/x_train.p', 'wb') as f:

        pickle.dump(x_train[:2000000], f)

    with open('pg_data/SAD/y_train.p', 'wb') as f:

        pickle.dump(y_train[:2000000], f)

    with open('pg_data/SAD/x_val.p', 'wb') as f:

        pickle.dump(x_val[:250000], f)

    with open('pg_data/SAD/y_val.p', 'wb') as f:

        pickle.dump(y_val[:250000], f)







#create_all_data()












'''
plt.figure()

for i in range(10000):

    plt.clf()

    plt.plot(spect_chops[i])
    plt.pause(0.1)

    print(trunc_labels[i])


plt.show()
'''


'''
plt.figure()
plt.plot(labelz[:800]*500, c = 'r')


plt.figure()
plt.pcolormesh(t[:800], f[:60], np.abs(S[:60,:800]))
#plt.plot(labelz[:800]*500, c = 'r')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
'''
