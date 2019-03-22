import matplotlib.pyplot as plt
import numpy as np
import random
import h5py
import pickle
from data_utils import *

import tensorflow as tf
#from tensorflow.keras.layers import *

hf = h5py.File('SAD_data/1D_dataset.h5', 'r')

d = hf['d_series']['d_set_1']
l = hf['labels']['labels_1']

batch_size = 32
#truncated_backprop_length = 100
seq_length = 300

n_hidden = 128

model = tf.keras.Sequential()

model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_hidden, return_sequences=True), input_shape=(seq_length,1)))
#model.add(Bidirectional(LSTM(10)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_hidden, return_sequences=True)))

model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='relu')))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid')))
#model.add(tf.keras.layers.Dense(seq_length, activation='linear'))
# Take a look at the model summary
model.summary()

#model.add(Dense(64, activation='linear'))#'softmax'))
#model.add(Activation('softmax'))
model.compile(loss='mse', optimizer='adam')

def get_batch(seq_length, batch_size, seq = d, labels = l):

	x = []
	l = []

	max_ind = seq.shape[0] - seq_length

	for s in range(batch_size):

		start_ind = random.randint(0, max_ind - 1)

		xs = seq[start_ind : start_ind + seq_length]
		ls = labels[start_ind : start_ind + seq_length]

		x.append(xs)
		l.append(ls)

	x = np.expand_dims(np.array(x), -1)
	l = np.expand_dims(np.array(l), -1)

	# sample weights

	return x, l


#print(d.shape)
#print(l.shape)

losses = []

plt.figure()

for e in range(2000):

	xb, yb = get_batch(seq_length, batch_size)

	#print(xb.shape)
	#print(yb.shape)


	print(e)
	hist = model.fit(xb, yb)
	#print(hist)
	losses.append(hist.history['loss'])

	plt.clf()
	plt.plot(losses)
	plt.ylim(0,0.1)
	plt.pause(0.01)


plt.show()

'''
for i in range(5):

	plt.figure()
	plt.plot(xb[i])
	plt.plot(yb[i]*50)
	plt.show()
'''

'''
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])


# Forward direction cell
lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
# Backward direction cell
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

outputs, output_state_fw, output_state_bw = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, batchX_placeholder)
'''


'''
print(d.shape)
print(l.shape)

plt.figure()
plt.plot(d[10000:11000])
plt.plot(l[10000:11000]*50)
plt.show()
'''
