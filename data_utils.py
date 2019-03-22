import re
from scipy.io.wavfile import read
from scipy import signal
import numpy as np
import h5py
import pickle
import random

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

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

    return f,t, np.abs(Sxx), sample_wav[1]

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

def generate_batches_from_hdf5(hdf5_fn, file_inds, batch_size = 16, spect_length = 100):

	mean_fn = 'SAD_data/mean.p'

	with open(mean_fn, 'rb') as f:

		means = pickle.load(f)

	with h5py.File(hdf5_fn, 'r') as hf:

		while 1:

			spects = []
			labels = []
			sample_weights = []

			for _ in range(batch_size):

				#fn_ind = random.randint(1,top_file_ind)

				fn_ind = random.choice(file_inds)

				s = hf['spects']['spect_' + str(fn_ind)]
				l = hf['labels']['labels_' + str(fn_ind)]

				max_len = s.shape[1]

				pos_ind = random.randint(0, max_len - spect_length - 1)

				spect = s[:, pos_ind : pos_ind + spect_length]
				label = l[pos_ind : pos_ind + spect_length]

				spect = (spect.transpose() - means).transpose()
				spect = np.expand_dims(spect, -1)

				sample_weight = 1.0 + 2.0 * np.mean(label)

				#print(sample_weight)

				spects.append(spect)
				labels.append(label)
				sample_weights.append(sample_weight)

			spects = np.array(spects)
			labels = np.array(labels)
			sample_weights = np.array(sample_weights)

			#print(spects.shape, labels.shape)

			yield (spects, labels, sample_weights)

def sequential_from_hdf5(hdf5_fn, fn_ind, spect_length = 100, nb_spects = 100, step_size = 50):

	mean_fn = 'SAD_data/mean.p'

	with open(mean_fn, 'rb') as f:

		means = pickle.load(f)

	with h5py.File(hdf5_fn, 'r') as hf:

		s = hf['spects']['spect_' + str(fn_ind)]
		l = hf['labels']['labels_' + str(fn_ind)]

		pos_ind = 0



		spects = []
		labels = []
		#sample_weights = []

		for _ in range(nb_spects):

			#fn_ind = random.randint(1,top_file_ind)

			#fn_ind = random.choice(file_inds)

			#max_len = s.shape[1]

			#pos_ind = random.randint(0, max_len - spect_length - 1)

			spect = s[:, pos_ind : pos_ind + spect_length]
			label = l[pos_ind : pos_ind + spect_length]

			spect = (spect.transpose() - means).transpose()
			spect = np.expand_dims(spect, -1)

			#sample_weight = 1.0 + 3.0 * np.mean(label)

			#print(sample_weight)

			spects.append(spect)
			labels.append(label)
			#sample_weights.append(sample_weight)

			pos_ind += step_size

		spects = np.array(spects)
		labels = np.array(labels)
		#sample_weights = np.array(sample_weights)

			#print(spects.shape, labels.shape)

		return spects, labels



'''
def generate_batches_from_hdf5(hdf5_fn, file_inds, batch_size = 16, spect_length = 100):

	mean_fn = 'SAD_data/mean.p'

	with open(mean_fn, 'rb') as f:

		means = pickle.load(f)

	with h5py.File(hdf5_fn, 'r') as hf:

		while 1:

			spects = []
			labels = []

			for _ in range(batch_size):

				#fn_ind = random.randint(1,top_file_ind)

				fn_ind = random.choice(file_inds)

				s = hf['spects']['spect_' + str(fn_ind)]
				l = hf['labels']['labels_' + str(fn_ind)]

				max_len = s.shape[1]

				pos_ind = random.randint(0, max_len - spect_length - 1)

				spect = s[:, pos_ind : pos_ind + spect_length]
				label = l[pos_ind : pos_ind + spect_length]

				spect = (spect.transpose() - means).transpose()
				spect = np.expand_dims(spect, -1)

				spects.append(spect)
				labels.append(label)

			spects = np.array(spects)
			labels = np.array(labels)

			#print(spects.shape, labels.shape)

			yield (spects, labels)
'''
