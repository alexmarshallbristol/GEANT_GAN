from numpy.random import seed
# seed(1)
from tensorflow import set_random_seed
# set_random_seed(2)

import numpy as np

import keras
from keras.layers import Input, Flatten, Dense, Reshape, Dropout, BatchNormalization, concatenate, merge, Lambda, Average, RepeatVector, Subtract
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import load_model, Model
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints, activations

import math

import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import argparse

from sklearn.ensemble import GradientBoostingClassifier

import scipy.stats as stats

# from ops import Dense3D
import ops

G_architecture = [1024,1024,1024,512,256,64,16,8]
D_architecture = [128,64,32,16,8,4]

number_of_noise_dims = 5

# 0 - laptop, 1 - deepthought 2-bc
where_running = 1
test_number = 'test_1'

# Define training variables and settings ...

epochs = int(1E30)

# Number of samples see per training step
batch = 10
test_batch = 1000
# Number of training steps between save_intervals
save_interval = 2500


_EPSILON = K.epsilon()

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)


parser = argparse.ArgumentParser()

parser.add_argument('-l', action='store', dest='learning_rate', type=float,
					help='learning rate', default=0.00002)

results = parser.parse_args()


# Define optimizers ...

optimizerG = Adam(lr=results.learning_rate, beta_1=0.5, decay=0, amsgrad=True)
optimizerD = Adam(lr=results.learning_rate, beta_1=0.5, decay=0, amsgrad=True)


# Build Generative model ...

noise = Input(shape=(1,number_of_noise_dims))

initial_state = Input(shape=(1,1))

merge_inputs = merge([noise,initial_state], mode='concat', concat_axis=2)

H = Dense(int(G_architecture[0]))(merge_inputs)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

for layer in G_architecture[1:]:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = BatchNormalization(momentum=0.8)(H)


H = Dense(6, activation='tanh')(H)
final_state_guess = Reshape((1,6))(H)

g_output = concatenate([initial_state, final_state_guess],axis=2)

generator = Model(inputs=[noise,initial_state], outputs=[g_output])

generator.compile(loss=_loss_generator, optimizer=optimizerG)
generator.summary()



# Build Discriminator model ...

d_input = Input(shape=(1,8))

H = Flatten()(d_input)

for layer in D_architecture:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)

d_output = Dense(1, activation='sigmoid')(H)

discriminator = Model(d_input, d_output)

discriminator.compile(loss='binary_crossentropy',optimizer=optimizerD)
# discriminator.compile(loss=_binary_crossentropy,optimizer=optimizerD)
discriminator.summary()





def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

make_trainable(discriminator, False)

initial_state_w_noise = Input(shape=(1,number_of_noise_dims))

inital_state = Input(shape=(1,1))

H = generator([initial_state_w_noise,inital_state])

# add a dimension to H
def add_dims_lambda(input_array):

	input_array_pre_mom = input_array[:,:,0]
	input_array_post_mom_x = input_array[:,:,4]
	input_array_post_mom_y = input_array[:,:,5]
	input_array_post_mom_z = input_array[:,:,6]

	input_array_pre_mom = (input_array_pre_mom/2) + 0.5
	input_array_pre_mom = input_array_pre_mom * 400
	input_array_pre_mom = input_array_pre_mom + 0

	input_array_post_mom_x = (input_array_post_mom_x/2) + 0.5
	input_array_post_mom_x = input_array_post_mom_x * 0.5
	input_array_post_mom_x = input_array_post_mom_x - 0.25

	input_array_post_mom_y = (input_array_post_mom_y/2) + 0.5
	input_array_post_mom_y = input_array_post_mom_y * 0.5
	input_array_post_mom_y = input_array_post_mom_y - 0.25

	input_array_post_mom_z = (input_array_post_mom_z/2) + 0.5
	input_array_post_mom_z = input_array_post_mom_z * 400
	input_array_post_mom_z = input_array_post_mom_z + 0

	out_mom = K.sqrt(K.square(input_array_post_mom_x) + K.square(input_array_post_mom_y) + K.square(input_array_post_mom_z))

	mom_diff = input_array_pre_mom - out_mom

	mom_diff = mom_diff - 0

	mom_diff = mom_diff/400

	mom_diff = (mom_diff - 0.5) * 2

	mom_diff = K.expand_dims(mom_diff, 1)

	input_array = K.concatenate([input_array, mom_diff],axis=2)

	return input_array

def add_dims_lambda_output_shape(input_shape):
	shape = list(input_shape)
	shape[2] += 1
	return tuple(shape)

add_dims_layer = Lambda(add_dims_lambda,
								  output_shape=add_dims_lambda_output_shape)
H = add_dims_layer(H)

gan_output = discriminator(H)

GAN_stacked = Model(inputs=[initial_state_w_noise,inital_state], outputs=[gan_output])
GAN_stacked.compile(loss=_loss_generator, optimizer=optimizerD)
GAN_stacked.summary()





# Define arrays to save loss data ...

d_loss_list = g_loss_list = np.empty((0,2))



if where_running == 0:
	training_data_location = '/Users/am13743/Desktop/style-transfer-GANs/FINITE_ELEMENT_GEANT/data/'
	output_location = '/Users/am13743/Desktop/style-transfer-GANs/FINITE_ELEMENT_GEANT/data/plots/'
elif where_running == 1:
	training_data_location = ''
	output_location = ''
elif where_running == 2:
	training_data_location = '/mnt/storage/scratch/am13743/GEANT_GAN/'
	output_location = '/mnt/storage/scratch/am13743/GEANT_GAN/%s/'%test_number


# Start training loop ...

FairSHiP_sample = np.load('%ssimulated_array_pre_processed_broaden.npy'%(training_data_location))

print('Loading new file - new file has',np.shape(FairSHiP_sample)[0],'samples from FairSHiP.')

split = [0.9,0.1]

print('Split:',split)
split_index = int(split[0]*np.shape(FairSHiP_sample)[0])
list_for_np_choice = np.arange(np.shape(FairSHiP_sample)[0]) 
random_indicies = np.random.choice(list_for_np_choice, size=np.shape(FairSHiP_sample)[0], replace=False)
training_sample = FairSHiP_sample[:split_index]
test_sample = FairSHiP_sample[split_index:]

print('Training:',np.shape(training_sample), 'Test:',np.shape(test_sample))

list_for_np_choice = np.arange(np.shape(training_sample)[0]) 
list_for_np_choice_test = np.arange(np.shape(test_sample)[0]) 


for e in range(epochs):


	random_indicies = np.random.choice(list_for_np_choice, size=(3,batch), replace=False)

	# Prepare training samples for D ...

	d_real_training = training_sample[random_indicies[0]]
	d_fake_training = training_sample[random_indicies[1]]

	random_dimension = np.random.uniform(-1,1,(batch,number_of_noise_dims))


	d_fake_training_initial, throw_away = np.split(d_fake_training, [1], axis=1) 

	random_dimension = np.expand_dims(random_dimension,1)
	d_fake_training_initial = np.expand_dims(d_fake_training_initial,1)

	synthetic_output = generator.predict([random_dimension, d_fake_training_initial]) # Run initial muon parameters through G for a final state guess and initial state in shape (2,5)


	legit_labels = np.ones((int(batch), 1)) # Create label arrays
	gen_labels = np.zeros((int(batch), 1))


	d_real_training = np.expand_dims(d_real_training,1)

	d_real_training_real_values = ops.propagate_post_process_2(d_real_training)
	synthetic_output_real_values = ops.propagate_post_process_2(synthetic_output)

	d_real_training_mom_difference = d_real_training_real_values[:,:,0] - np.sqrt(np.add(np.add(d_real_training_real_values[:,:,4]**2,d_real_training_real_values[:,:,5]**2),d_real_training_real_values[:,:,6]**2))
	synthetic_output_mom_difference = synthetic_output_real_values[:,:,0] - np.sqrt(np.add(np.add(synthetic_output_real_values[:,:,4]**2,synthetic_output_real_values[:,:,5]**2),synthetic_output_real_values[:,:,6]**2))

	d_real_training_mom_difference = np.expand_dims(d_real_training_mom_difference, 1)
	synthetic_output_mom_difference = np.expand_dims(synthetic_output_mom_difference, 1)
	# print(np.shape(d_real_training), np.shape(d_real_training_mom_difference))

	d_real_training = np.append(d_real_training, d_real_training_mom_difference,axis=2)
	synthetic_output = np.append(synthetic_output, synthetic_output_mom_difference,axis=2)

	d_loss_legit = discriminator.train_on_batch(d_real_training, legit_labels) # Train D
	d_loss_gen = discriminator.train_on_batch(synthetic_output, gen_labels)


	random_dimension = np.random.uniform(-1,1,(batch,number_of_noise_dims))

	g_training = np.random.uniform(-1,1,(batch,1))

	y_mislabled = np.zeros((batch, 1))

	random_dimension = np.expand_dims(random_dimension,1)
	g_training = np.expand_dims(g_training,1)


	g_loss = GAN_stacked.train_on_batch([random_dimension, g_training], [y_mislabled])

	d_loss_list = np.append(d_loss_list, [[e,(d_loss_legit+d_loss_gen)/2]], axis=0)
	g_loss_list = np.append(g_loss_list, [[e, g_loss]], axis=0)


	if e % 100 == 0 and e > 1: 
		print('Step:',e)

	if e % save_interval == 0 and e > 1: 

		print('Saving',e,'...')

		plt.subplot(1,2,1)
		plt.title('Discriminator loss')
		plt.plot(d_loss_list[:,0],d_loss_list[:,1])
		plt.subplot(1,2,2)
		plt.title('Generator loss')
		plt.plot(g_loss_list[:,0],g_loss_list[:,1])
		plt.savefig('%sloss.png'%output_location,bbox_inches='tight')
		plt.close('all')

		# Create test samples ... 

		random_indicies = np.random.choice(list_for_np_choice_test, size=test_batch, replace=False)

		sample_to_test = test_sample[random_indicies]

		random_dimension = np.random.uniform(-1,1,(test_batch,number_of_noise_dims))

		sample_to_test_initial, throw_away = np.split(sample_to_test, [1], axis=1) # Remove the real final state information

		sample_to_test_initial = np.expand_dims(sample_to_test_initial,1)

		random_dimension = np.expand_dims(random_dimension,1)

		g_test = np.random.uniform(1,-1,(test_batch,1))
		g_test = np.expand_dims(g_test,1)
		synthetic_test_output = generator.predict([random_dimension, g_test]) # Run initial muon parameters through G for a final state guess and initial state in shape (2,5)

		print('GEANT4 test sample shape:',np.shape(sample_to_test))
		print('GAN test sample shape:',np.shape(synthetic_test_output))

		
		for first in range(0, 7):
			ops.plot_1d_hists(first, sample_to_test, synthetic_test_output, output_location, e)

		number_of_muons_sim = 2500
		GAN_steps_sim = 10

		ops.propagate_z_distance(number_of_muons_sim, GAN_steps_sim, 0, output_location, e, generator, number_of_noise_dims)
		ops.propagate_z_distance(number_of_muons_sim, GAN_steps_sim, 1, output_location, e, generator, number_of_noise_dims)
		ops.propagate_z_distance(number_of_muons_sim, GAN_steps_sim, 2, output_location, e, generator, number_of_noise_dims)

		ops.plot_bethe_bloch(100,10, output_location, generator, number_of_noise_dims)

		generator.save('%sGenerator.h5'%output_location)

		print('Saving complete...')


