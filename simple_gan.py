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


G_architecture = [2048,1024,512,256,64,32,16,8]
D_architecture = [256,128,64,32,16,8,4]

number_of_noise_dims = 10

# 0 - laptop, 1 - deepthought 2-bc
where_running = 0
test_number = 'test_11'

_EPSILON = K.epsilon() # 10^-7 by default. Epsilon is used as a small constant to avoid ever dividing by zero. 
# _EPSILON = 1E-15
def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)


class Dense3D(Layer):

	"""
	A 3D, trainable, dense tensor product layer
	"""

	def __init__(self, first_dim, last_dim, init='glorot_uniform',
				 activation=None, weights=None,
				 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True, input_dim=None, **kwargs):

		self.init = initializers.get(init)
		self.activation = activations.get(activation)

		self.input_dim = input_dim
		self.first_dim = first_dim
		self.last_dim = last_dim

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		self.initial_weights = weights
		self.input_spec = [InputSpec(ndim=2)]

		if self.input_dim:
			kwargs['input_shape'] = (self.input_dim,)
		super(Dense3D, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 2
		input_dim = input_shape[1]
		self.input_spec = [InputSpec(dtype=K.floatx(),
									 shape=(None, input_dim))]

		self.W = self.add_weight(
			(self.first_dim, input_dim, self.last_dim),
			initializer=self.init,
			name='{}_W'.format(self.name),
			regularizer=self.W_regularizer,
			constraint=self.W_constraint
		)
		if self.bias:
			self.b = self.add_weight(
				(self.first_dim, self.last_dim),
				initializer='zero',
				name='{}_b'.format(self.name),
				regularizer=self.b_regularizer,
				constraint=self.b_constraint
			)
		else:
			self.b = None

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights
		self.built = True

	def call(self, x, mask=None):
		out = K.reshape(K.dot(x, self.W), (-1, self.first_dim, self.last_dim))
		if self.bias:
			out += self.b
		return self.activation(out)

	def get_output_shape_for(self, input_shape):
		assert input_shape and len(input_shape) == 2
		return (input_shape[0], self.first_dim, self.last_dim)

	def get_config(self):
		config = {
			'first_dim': self.first_dim,
			'last_dim': self.last_dim,
			'init': self.init.__name__,
			'activation': self.activation.__name__,
			'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
			'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
			'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
			'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
			'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
			'bias': self.bias,
			'input_dim': self.input_dim
		}
		base_config = super(Dense3D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


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

d_input = Input(shape=(1,7))

H = Flatten()(d_input)

for layer in D_architecture:

	H = Dense(int(layer))(H)
	H = LeakyReLU(alpha=0.2)(H)
	H = Dropout(0.2)(H)

######################

# K_x = Dense3D(8, 10)(H)
# minibatch_featurizer = Lambda(minibatch_discriminator,
#                                   output_shape=minibatch_output_shape)
# features = merge([
#         minibatch_featurizer(K_x),
#         H
#     ], mode='concat')

######################

######################
# only x and y minibatch
######################
# def split_tensor(input_tensor):
# 	out = input_tensor[:,:,1:3] # mini-batch stuff only over x and y 
# 	return out

# def split_tensor_output_shape(input_shape):
# 	shape = list(input_shape)
# 	assert len(shape) == 3  
# 	shape[2] = 2
# 	return tuple(shape)

# split_layer = Lambda(split_tensor, split_tensor_output_shape)
# # print(K.int_shape(d_input),' minibatch - d_input')
# x_y_input = split_layer(d_input)

# # print(K.int_shape(x_y_input),' minibatch - x_y_input')
# x_y_input = Flatten()(x_y_input)

# # print(K.int_shape(x_y_input),' minibatch - x_y_input')
# K_x = Dense3D(2, 5)(x_y_input)

# # print(K.int_shape(K_x),' minibatch - K_x')
# def minibatch_discriminator(x):
# 	""" Computes minibatch discrimination features from input tensor x FROM LAGANS PAPER"""
# 	diffs = K.expand_dims(x, 3) - \
# 		K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
# 	l1_norm = K.sum(K.abs(diffs), axis=2)
# 	return K.sum(K.exp(-l1_norm), axis=2)


# def minibatch_output_shape(input_shape):
# 	""" Computes output shape for a minibatch discrimination layer"""
# 	shape = list(input_shape)
# 	assert len(shape) == 2  # only valid for 3D tensors
# 	return tuple(shape[:2])

# minibatch_featurizer = Lambda(minibatch_discriminator,
# 								  output_shape=minibatch_output_shape)
# minibatch_featurizer_out = minibatch_featurizer(K_x)
# # print(K.int_shape(minibatch_featurizer_out),' minibatch - minibatch_featurizer_out')
# J = Dense(4)(minibatch_featurizer_out)
# # print(K.int_shape(J),' minibatch - J')

# ######################
# ######################

# ######################


# features = merge([
# 		J,
# 		H,
# 	], mode='concat')

######################

# H = Dense(32)(H)
# H = LeakyReLU(alpha=0.2)(H)
# H = Dropout(0.2)(H)

# H = Dense(16)(H)
# H = LeakyReLU(alpha=0.2)(H)
# H = Dropout(0.2)(H)

# H = Dense(8)(H)
# H = LeakyReLU(alpha=0.2)(H)
# H = Dropout(0.2)(H)

# H = Dense(4)(H)
# H = LeakyReLU(alpha=0.2)(H)
# H = Dropout(0.2)(H)

d_output = Dense(1, activation='sigmoid')(H)

discriminator = Model(d_input, d_output)

discriminator.compile(loss='binary_crossentropy',optimizer=optimizerD)
discriminator.summary()


# quit()

# Build stacked GAN model ...

def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

make_trainable(discriminator, False)

initial_state_w_noise = Input(shape=(1,number_of_noise_dims))

inital_state = Input(shape=(1,1))

H = generator([initial_state_w_noise,inital_state])

# add a dimension to H
# def add_dims_lambda(input_array):

# 	#1
# 	ones = K.ones((K.shape(input_array)[0],K.shape(input_array)[1],1))
# 	mean_x = K.squeeze(K.mean(input_array[:,:,1],keepdims=True),0)
# 	ones = ones * mean_x
# 	input_array = K.concatenate([input_array, ones],axis=2)

# 	#2
# 	ones = K.ones((K.shape(input_array)[0],K.shape(input_array)[1],1))
# 	mean_y = K.squeeze(K.mean(input_array[:,:,2],keepdims=True),0)
# 	ones = ones * mean_y
# 	input_array = K.concatenate([input_array, ones],axis=2)

# 	#3
# 	ones = K.ones((K.shape(input_array)[0],K.shape(input_array)[1],1))
# 	mean = K.squeeze(K.std(input_array[:,:,1],keepdims=True),0)
# 	ones = ones * mean
# 	input_array = K.concatenate([input_array, ones],axis=2)

# 	#4
# 	ones = K.ones((K.shape(input_array)[0],K.shape(input_array)[1],1))
# 	mean = K.squeeze(K.std(input_array[:,:,2],keepdims=True),0)
# 	ones = ones * mean
# 	input_array = K.concatenate([input_array, ones],axis=2)

# 	#

# 	#5
# 	x_diff = input_array[:,:,1] - mean_x
# 	y_diff = input_array[:,:,2] - mean_y

# 	x_diff2 = x_diff * x_diff
# 	y_diff2 = y_diff * y_diff

# 	x_diff_y_diff = x_diff * y_diff

# 	sum_x_diff_y_diff = K.sum(x_diff_y_diff, keepdims=True)
# 	sum_x_diff2 = K.sum(x_diff2, keepdims=True)
# 	sum_y_diff2 = K.sum(y_diff2, keepdims=True)

# 	sqrt_sum_x_diff2 = K.sqrt(sum_x_diff2)
# 	sqrt_sum_y_diff2 = K.sqrt(sum_y_diff2)

# 	sum_x_diff_y_diff = K.squeeze(sum_x_diff_y_diff,0)
# 	sqrt_sum_x_diff2 = K.squeeze(sqrt_sum_x_diff2,0)
# 	sqrt_sum_y_diff2 = K.squeeze(sqrt_sum_y_diff2,0)

# 	r = sum_x_diff_y_diff/(sqrt_sum_x_diff2*sqrt_sum_y_diff2)

# 	ones = K.ones((K.shape(input_array)[0],K.shape(input_array)[1],1))
# 	ones = ones * r
# 	input_array = K.concatenate([input_array, ones],axis=2)

# 	#

# 	#6
# 	mom_diff = input_array[:,:,0] - input_array[:,:,6]

# 	mean_mom_diff = K.squeeze(K.mean(mom_diff,keepdims=True),0)
# 	ones = ones * mean_mom_diff
# 	input_array = K.concatenate([input_array, ones],axis=2)


# 	# min and max values?

# 	#7
# 	ones = K.ones((K.shape(input_array)[0],K.shape(input_array)[1],1))
# 	min_x = K.squeeze(K.min(input_array[:,:,1],keepdims=True),0)
# 	ones = ones * min_x
# 	input_array = K.concatenate([input_array, ones],axis=2)

# 	#8
# 	ones = K.ones((K.shape(input_array)[0],K.shape(input_array)[1],1))
# 	max_x = K.squeeze(K.max(input_array[:,:,1],keepdims=True),0)
# 	ones = ones * max_x
# 	input_array = K.concatenate([input_array, ones],axis=2)

# 	#9
# 	ones = K.ones((K.shape(input_array)[0],K.shape(input_array)[1],1))
# 	min_y = K.squeeze(K.min(input_array[:,:,2],keepdims=True),0)
# 	ones = ones * min_y
# 	input_array = K.concatenate([input_array, ones],axis=2)

# 	#10
# 	ones = K.ones((K.shape(input_array)[0],K.shape(input_array)[1],1))
# 	max_y = K.squeeze(K.max(input_array[:,:,2],keepdims=True),0)
# 	ones = ones * max_y
# 	input_array = K.concatenate([input_array, ones],axis=2)

# 	return input_array

# def add_dims_lambda_output_shape(input_shape):
# 	shape = list(input_shape)
# 	shape[2] += 10
# 	return tuple(shape)

# add_dims_layer = Lambda(add_dims_lambda,
								  # output_shape=add_dims_lambda_output_shape)
# H = add_dims_layer(H)

gan_output = discriminator(H)

GAN_stacked = Model(inputs=[initial_state_w_noise,inital_state], outputs=[gan_output])
GAN_stacked.compile(loss=_loss_generator, optimizer=optimizerD)
GAN_stacked.summary()



# Define arrays to save loss data ...

d_loss_list = g_loss_list = np.empty((0,2))



# Define training variables and settings ...

epochs = int(1E30)

# Number of samples see per training step
batch = 50
test_batch = 1000
# Number of training steps between save_intervals
save_interval = 1000


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
	# quit()
	






	random_dimension = np.random.uniform(-1,1,(batch,number_of_noise_dims))


	d_fake_training_initial, throw_away = np.split(d_fake_training, [1], axis=1) 

	# d_fake_training_initial_w_rand = np.concatenate((d_fake_training_initial, random_dimension),axis=1) # Add dimension of random noise

	# d_fake_training_initial_w_rand = np.expand_dims(d_fake_training_initial_w_rand,1)
	random_dimension = np.expand_dims(random_dimension,1)
	d_fake_training_initial = np.expand_dims(d_fake_training_initial,1)

	synthetic_output = generator.predict([random_dimension, d_fake_training_initial]) # Run initial muon parameters through G for a final state guess and initial state in shape (2,5)


	legit_labels = np.ones((int(batch), 1)) # Create label arrays
	gen_labels = np.zeros((int(batch), 1))


	d_real_training = np.expand_dims(d_real_training,1)


	# # HERE CALCULATE AND APPEND IN THE MEAN AND STD ETC
	# d_real_training = np.append(d_real_training, np.ones((batch,1,1))*np.mean(d_real_training[:,:,1]),axis=2)
	# synthetic_output = np.append(synthetic_output, np.ones((batch,1,1))*np.mean(synthetic_output[:,:,1]),axis=2)

	# d_real_training = np.append(d_real_training, np.ones((batch,1,1))*np.mean(d_real_training[:,:,2]),axis=2)
	# synthetic_output = np.append(synthetic_output, np.ones((batch,1,1))*np.mean(synthetic_output[:,:,2]),axis=2)

	# d_real_training = np.append(d_real_training, np.ones((batch,1,1))*np.std(d_real_training[:,:,1]),axis=2)
	# synthetic_output = np.append(synthetic_output, np.ones((batch,1,1))*np.std(synthetic_output[:,:,1]),axis=2)

	# d_real_training = np.append(d_real_training, np.ones((batch,1,1))*np.std(d_real_training[:,:,2]),axis=2)
	# synthetic_output = np.append(synthetic_output, np.ones((batch,1,1))*np.std(synthetic_output[:,:,2]),axis=2)

	# d_real_training = np.append(d_real_training, np.ones((batch,1,1))*np.corrcoef([np.squeeze(d_real_training[:,:,1]),np.squeeze(d_real_training[:,:,2])])[0][1],axis=2)
	# synthetic_output = np.append(synthetic_output, np.ones((batch,1,1))*np.corrcoef([np.squeeze(synthetic_output[:,:,1]),np.squeeze(synthetic_output[:,:,2])])[0][1],axis=2)

	# d_real_training = np.append(d_real_training, np.ones((batch,1,1))*np.mean(np.subtract(d_real_training[:,:,0],d_real_training[:,:,6])),axis=2)
	# synthetic_output = np.append(synthetic_output, np.ones((batch,1,1))*np.mean(np.subtract(synthetic_output[:,:,0],synthetic_output[:,:,6])),axis=2)



	# d_real_training = np.append(d_real_training, np.ones((batch,1,1))*np.amin(d_real_training[:,:,1]),axis=2)
	# synthetic_output = np.append(synthetic_output, np.ones((batch,1,1))*np.amin(synthetic_output[:,:,1]),axis=2)

	# d_real_training = np.append(d_real_training, np.ones((batch,1,1))*np.amax(d_real_training[:,:,1]),axis=2)
	# synthetic_output = np.append(synthetic_output, np.ones((batch,1,1))*np.amax(synthetic_output[:,:,1]),axis=2)


	# d_real_training = np.append(d_real_training, np.ones((batch,1,1))*np.amin(d_real_training[:,:,2]),axis=2)
	# synthetic_output = np.append(synthetic_output, np.ones((batch,1,1))*np.amin(synthetic_output[:,:,2]),axis=2)

	# d_real_training = np.append(d_real_training, np.ones((batch,1,1))*np.amax(d_real_training[:,:,2]),axis=2)
	# synthetic_output = np.append(synthetic_output, np.ones((batch,1,1))*np.amax(synthetic_output[:,:,2]),axis=2)



	d_loss_legit = discriminator.train_on_batch(d_real_training, legit_labels) # Train D
	d_loss_gen = discriminator.train_on_batch(synthetic_output, gen_labels)



	random_dimension = np.random.uniform(-1,1,(batch,number_of_noise_dims))

	g_training = np.random.uniform(-1,1,(batch,1))
	# g_training_w_noise = np.concatenate((g_training, random_dimension),axis=1) # Add dimension of random noise

	y_mislabled = np.zeros((batch, 1))

	# g_training_w_noise = np.expand_dims(g_training_w_noise,1)
	random_dimension = np.expand_dims(random_dimension,1)
	g_training = np.expand_dims(g_training,1)

	g_loss = GAN_stacked.train_on_batch([random_dimension, g_training], y_mislabled)

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

		# sample_to_test_initial_w_rand = np.concatenate((sample_to_test_initial, random_dimension),axis=1) # Add dimension of random noise

		sample_to_test_initial = np.expand_dims(sample_to_test_initial,1)
		# sample_to_test_initial_w_rand = np.expand_dims(sample_to_test_initial_w_rand,1)
		random_dimension = np.expand_dims(random_dimension,1)

		g_test = np.random.uniform(1,-1,(test_batch,1))
		g_test = np.expand_dims(g_test,1)
		synthetic_test_output = generator.predict([random_dimension, g_test]) # Run initial muon parameters through G for a final state guess and initial state in shape (2,5)

		print('GEANT4 test sample shape:',np.shape(sample_to_test))
		print('GAN test sample shape:',np.shape(synthetic_test_output))




		def plot_1d_hists(dim):

			plt.figure(figsize=(8,8))

			plt.subplot(2,2,1)
			plt.hist([sample_to_test[:,dim],synthetic_test_output[:,0,dim]],histtype='step', bins=50,label=['GEANT4','GAN'])
			plt.legend(loc='upper right')
			plt.tick_params(axis='y', which='both', labelsize=5)
			plt.tick_params(axis='x', which='both', labelsize=5)

			plt.subplot(2,2,2)
			plt.hist([sample_to_test[:,dim],synthetic_test_output[:,0,dim]],histtype='step', bins=50,label=['GEANT4','GAN'])
			plt.legend(loc='upper right')
			plt.yscale('log')
			plt.tick_params(axis='y', which='both', labelsize=5)
			plt.tick_params(axis='x', which='both', labelsize=5)

			plt.subplot(2,2,3)
			plt.hist([sample_to_test[:,dim],synthetic_test_output[:,0,dim]],histtype='step', bins=50,label=['GEANT4','GAN'],range=[-1,1])
			plt.legend(loc='upper right')
			plt.tick_params(axis='y', which='both', labelsize=5)
			plt.tick_params(axis='x', which='both', labelsize=5)

			plt.subplot(2,2,4)
			plt.hist([sample_to_test[:,dim],synthetic_test_output[:,0,dim]],histtype='step', bins=50,label=['GEANT4','GAN'],range=[-1,1])
			plt.legend(loc='upper right')
			plt.yscale('log')
			plt.tick_params(axis='y', which='both', labelsize=5)
			plt.tick_params(axis='x', which='both', labelsize=5)

			plt.savefig('%shist_%d.png'%(output_location,dim),bbox_inches='tight')
			plt.savefig('%s%d/hist_%d_%d.png'%(output_location,dim,dim,e),bbox_inches='tight')

			plt.close('all')

		for first in range(0, 7):
			plot_1d_hists(first)
		# quit()





		def propagate_z_distance(number_of_muons,GAN_steps):

			def preprocess_pz(input_array, minimum, full_range):
				input_array += minimum * (-1)
				input_array = input_array/full_range

				return input_array

			def unpreprocess_pz(input_array, minimum, full_range):
				# input_array += minimum * (-1)
				# input_array = input_array/full_range

				input_array = input_array*full_range - minimum

				return input_array

			def propagate_post_process(gan_array):

				def propagate_post_process_inner_un_sqrt(input_array, power):

					signs = np.sign(input_array)

					input_array = np.absolute(input_array)

					input_array = np.power(input_array, 1/power)

					input_array = np.multiply(input_array, signs)

					return input_array

				gan_array[:,1] = propagate_post_process_inner_un_sqrt(gan_array[:,1], 0.3)
				gan_array[:,2] = propagate_post_process_inner_un_sqrt(gan_array[:,2], 0.3)


				gan_array = (gan_array/2) + 0.5

				def propagate_post_process_inner(input_array, minimum, full_range):

					input_array = input_array * full_range
					input_array += minimum

					return input_array

				for array in [gan_array]:
					array[:,0] = propagate_post_process_inner(array[:,0], 0, 400)
					array[:,1] = propagate_post_process_inner(array[:,1], -1.757/2, 1.757)
					array[:,2] = propagate_post_process_inner(array[:,2], -1.757/2, 1.757)
					array[:,3] = propagate_post_process_inner(array[:,3], 0, 1.757)
					array[:,4] = propagate_post_process_inner(array[:,4], -0.25, 0.5)
					array[:,5] = propagate_post_process_inner(array[:,5], -0.25, 0.5)
					array[:,6] = propagate_post_process_inner(array[:,6], 0, 400)

				return gan_array



			inital_x = np.zeros(number_of_muons)
			inital_y = np.zeros(number_of_muons)
			inital_z = np.zeros(number_of_muons)

			inital_px = np.zeros(number_of_muons)
			inital_py = np.zeros(number_of_muons)
			inital_pz = np.random.uniform(0,400,number_of_muons)


			initial_kinematics = np.swapaxes([inital_x,inital_y,inital_z,inital_px,inital_py,inital_pz],0,1)

			history = np.empty((0,number_of_muons,6))

			history = np.append(history,[initial_kinematics],axis=0)

			for i in range(0, GAN_steps):

				mom_in = np.sqrt(np.add(np.add(initial_kinematics[:,3]**2,initial_kinematics[:,4]**2),initial_kinematics[:,5]**2))
				
				pz_input = preprocess_pz(mom_in,0,400)

				pz_input = (np.subtract(pz_input,0.5)) * 2

				pz_input = np.expand_dims(pz_input,1)
				pz_input = np.expand_dims(pz_input,1)


				random_dimension = np.random.uniform(-1,1,(number_of_muons,10))
				random_dimension = np.expand_dims(random_dimension,1)

				# print(np.shape(pz_input))

				gan_output = np.squeeze(generator.predict([random_dimension,pz_input]))



				gan_output = propagate_post_process(gan_output)


				# print('gan_output',gan_output[0])


				theta_x = np.arctan(np.divide(-initial_kinematics[:,3],initial_kinematics[:,5]))

				mom_after_rotation_of_theta_x = np.swapaxes([initial_kinematics[:,3]*np.cos(theta_x)+initial_kinematics[:,5]*np.sin(theta_x),initial_kinematics[:,4],-initial_kinematics[:,3]*np.sin(theta_x)+initial_kinematics[:,5]*np.cos(theta_x)],0,1)

				theta_y = np.arctan(np.divide(mom_after_rotation_of_theta_x[:,1],mom_after_rotation_of_theta_x[:,2]))

				# mom_after_rotation_of_theta_x_and_theta_y = np.swapaxes([mom_after_rotation_of_theta_x[:,0],mom_after_rotation_of_theta_x[:,1]*np.cos(theta_y)-mom_after_rotation_of_theta_x[:,2]*np.sin(theta_y),mom_after_rotation_of_theta_x[:,1]*np.sin(theta_y)+mom_after_rotation_of_theta_x[:,2]*np.cos(theta_y)],0,1)



				gan_mom = np.swapaxes([gan_output[:,4],gan_output[:,5],gan_output[:,6]],0,1)
				gan_pos = np.swapaxes([gan_output[:,1],gan_output[:,2],gan_output[:,3]],0,1)

				gan_mom_rotate_back_theta_y = np.swapaxes([gan_mom[:,0],gan_mom[:,1]*np.cos(-theta_y)-gan_mom[:,2]*np.sin(-theta_y),gan_mom[:,1]*np.sin(-theta_y)+gan_mom[:,2]*np.cos(-theta_y)],0,1)
				gan_mom_forward_frame = np.swapaxes([gan_mom_rotate_back_theta_y[:,0]*np.cos(-theta_x)+gan_mom_rotate_back_theta_y[:,2]*np.sin(-theta_x),gan_mom_rotate_back_theta_y[:,1],-gan_mom_rotate_back_theta_y[:,0]*np.sin(-theta_x)+gan_mom_rotate_back_theta_y[:,2]*np.cos(-theta_x)],0,1)

				gan_pos_rotate_back_theta_y = np.swapaxes([gan_pos[:,0],gan_pos[:,1]*np.cos(-theta_y)-gan_pos[:,2]*np.sin(-theta_y),gan_pos[:,1]*np.sin(-theta_y)+gan_pos[:,2]*np.cos(-theta_y)],0,1)
				gan_pos_forward_frame = np.swapaxes([gan_pos_rotate_back_theta_y[:,0]*np.cos(-theta_x)+gan_pos_rotate_back_theta_y[:,2]*np.sin(-theta_x),gan_pos_rotate_back_theta_y[:,1],-gan_pos_rotate_back_theta_y[:,0]*np.sin(-theta_x)+gan_pos_rotate_back_theta_y[:,2]*np.cos(-theta_x)],0,1)





				final_kinematics = np.swapaxes([np.add(initial_kinematics[:,0],gan_pos_forward_frame[:,0]),
											np.add(initial_kinematics[:,1],gan_pos_forward_frame[:,1]),
												np.add(initial_kinematics[:,2],gan_pos_forward_frame[:,2]),
												gan_mom_forward_frame[:,0],
												gan_mom_forward_frame[:,1],
												gan_mom_forward_frame[:,2]],0,1)

				initial_kinematics = final_kinematics

				history = np.append(history,[initial_kinematics],axis=0)


			plot_particles = 10

			plt.figure(figsize=(8,8))

			plt.subplot(2,2,1)
			plt.plot(history[:,:plot_particles,2],history[:,:plot_particles,0])
			plt.xlabel('Z Coordinate (cm)')
			plt.ylabel('X Coordinate (cm)')

			plt.subplot(2,2,2)
			total_mom = np.sqrt(np.add(np.add(history[:,:,3]**2,history[:,:,4]**2),history[:,:,5]**2))
			plt.plot(history[:,:plot_particles,2],total_mom[:,:plot_particles])

			plt.xlabel('Z Coordinate (cm)')
			plt.ylabel('Total Momentum (GeV)')

			plt.subplot(2,2,3)
			plt.plot(history[:,:plot_particles,2],history[:,:plot_particles,1])
			plt.xlabel('Z Coordinate (cm)')
			plt.ylabel('Y Coordinate (cm)')

			plt.subplot(2,2,4)
			hist = np.histogram2d(history[-1,:number_of_muons_sim,0],history[-1,:number_of_muons_sim,1],bins=1,range=[[-0.25,0.25],[-0.25,0.25]])
			plt.text(-0.2,0.2,'%d/%d'%(np.sum(hist[0]),number_of_muons_sim))
			plt.hist2d(history[-1,:number_of_muons_sim,0],history[-1,:number_of_muons_sim,1],bins=100,norm=LogNorm(),range=[[-0.25,0.25],[-0.25,0.25]])
			plt.xlabel('X Coordinate (cm)')
			plt.ylabel('Y Coordinate (cm)')
			plt.colorbar()

			plt.savefig('%sprop/prop_%d.png'%(output_location,e), bbox_inches='tight')
			plt.savefig('%sprop.png'%output_location, bbox_inches='tight')

			return

		def propagate_z_distance_1_75(number_of_muons,GAN_steps):

			def preprocess_pz(input_array, minimum, full_range):
				input_array += minimum * (-1)
				input_array = input_array/full_range

				return input_array

			def unpreprocess_pz(input_array, minimum, full_range):
				# input_array += minimum * (-1)
				# input_array = input_array/full_range

				input_array = input_array*full_range - minimum

				return input_array

			def propagate_post_process(gan_array):

				def propagate_post_process_inner_un_sqrt(input_array, power):

					signs = np.sign(input_array)

					input_array = np.absolute(input_array)

					input_array = np.power(input_array, 1/power)

					input_array = np.multiply(input_array, signs)

					return input_array

				gan_array[:,1] = propagate_post_process_inner_un_sqrt(gan_array[:,1], 0.3)
				gan_array[:,2] = propagate_post_process_inner_un_sqrt(gan_array[:,2], 0.3)


				gan_array = (gan_array/2) + 0.5

				def propagate_post_process_inner(input_array, minimum, full_range):

					input_array = input_array * full_range
					input_array += minimum

					return input_array

				for array in [gan_array]:
					array[:,0] = propagate_post_process_inner(array[:,0], 0, 400)
					array[:,1] = propagate_post_process_inner(array[:,1], -1.757/2, 1.757)
					array[:,2] = propagate_post_process_inner(array[:,2], -1.757/2, 1.757)
					array[:,3] = propagate_post_process_inner(array[:,3], 0, 1.757)
					array[:,4] = propagate_post_process_inner(array[:,4], -0.25, 0.5)
					array[:,5] = propagate_post_process_inner(array[:,5], -0.25, 0.5)
					array[:,6] = propagate_post_process_inner(array[:,6], 0, 400)

				return gan_array



			inital_x = np.zeros(number_of_muons)
			inital_y = np.zeros(number_of_muons)
			inital_z = np.zeros(number_of_muons)

			inital_px = np.zeros(number_of_muons)
			inital_py = np.zeros(number_of_muons)
			inital_pz = np.random.uniform(0,400,number_of_muons)


			initial_kinematics = np.swapaxes([inital_x,inital_y,inital_z,inital_px,inital_py,inital_pz],0,1)

			history = np.empty((0,number_of_muons,6))

			history = np.append(history,[initial_kinematics],axis=0)

			for i in range(0, GAN_steps):

				mom_in = np.sqrt(np.add(np.add(initial_kinematics[:,3]**2,initial_kinematics[:,4]**2),initial_kinematics[:,5]**2))
				
				pz_input = preprocess_pz(mom_in,0,400)

				pz_input = (np.subtract(pz_input,0.5)) * 2

				pz_input = np.expand_dims(pz_input,1)
				pz_input = np.expand_dims(pz_input,1)


				random_dimension = np.random.uniform(-1,1,(number_of_muons,10))
				random_dimension = np.expand_dims(random_dimension,1)

				# print(np.shape(pz_input))

				gan_output = np.squeeze(generator.predict([random_dimension,pz_input]))



				gan_output = propagate_post_process(gan_output)


				gan_output[:,3] = np.ones(np.shape(gan_output[:,3]))*1.75
				# print('gan_output',gan_output[0])


				theta_x = np.arctan(np.divide(-initial_kinematics[:,3],initial_kinematics[:,5]))

				mom_after_rotation_of_theta_x = np.swapaxes([initial_kinematics[:,3]*np.cos(theta_x)+initial_kinematics[:,5]*np.sin(theta_x),initial_kinematics[:,4],-initial_kinematics[:,3]*np.sin(theta_x)+initial_kinematics[:,5]*np.cos(theta_x)],0,1)

				theta_y = np.arctan(np.divide(mom_after_rotation_of_theta_x[:,1],mom_after_rotation_of_theta_x[:,2]))

				# mom_after_rotation_of_theta_x_and_theta_y = np.swapaxes([mom_after_rotation_of_theta_x[:,0],mom_after_rotation_of_theta_x[:,1]*np.cos(theta_y)-mom_after_rotation_of_theta_x[:,2]*np.sin(theta_y),mom_after_rotation_of_theta_x[:,1]*np.sin(theta_y)+mom_after_rotation_of_theta_x[:,2]*np.cos(theta_y)],0,1)



				gan_mom = np.swapaxes([gan_output[:,4],gan_output[:,5],gan_output[:,6]],0,1)
				gan_pos = np.swapaxes([gan_output[:,1],gan_output[:,2],gan_output[:,3]],0,1)

				gan_mom_rotate_back_theta_y = np.swapaxes([gan_mom[:,0],gan_mom[:,1]*np.cos(-theta_y)-gan_mom[:,2]*np.sin(-theta_y),gan_mom[:,1]*np.sin(-theta_y)+gan_mom[:,2]*np.cos(-theta_y)],0,1)
				gan_mom_forward_frame = np.swapaxes([gan_mom_rotate_back_theta_y[:,0]*np.cos(-theta_x)+gan_mom_rotate_back_theta_y[:,2]*np.sin(-theta_x),gan_mom_rotate_back_theta_y[:,1],-gan_mom_rotate_back_theta_y[:,0]*np.sin(-theta_x)+gan_mom_rotate_back_theta_y[:,2]*np.cos(-theta_x)],0,1)

				gan_pos_rotate_back_theta_y = np.swapaxes([gan_pos[:,0],gan_pos[:,1]*np.cos(-theta_y)-gan_pos[:,2]*np.sin(-theta_y),gan_pos[:,1]*np.sin(-theta_y)+gan_pos[:,2]*np.cos(-theta_y)],0,1)
				gan_pos_forward_frame = np.swapaxes([gan_pos_rotate_back_theta_y[:,0]*np.cos(-theta_x)+gan_pos_rotate_back_theta_y[:,2]*np.sin(-theta_x),gan_pos_rotate_back_theta_y[:,1],-gan_pos_rotate_back_theta_y[:,0]*np.sin(-theta_x)+gan_pos_rotate_back_theta_y[:,2]*np.cos(-theta_x)],0,1)





				final_kinematics = np.swapaxes([np.add(initial_kinematics[:,0],gan_pos_forward_frame[:,0]),
											np.add(initial_kinematics[:,1],gan_pos_forward_frame[:,1]),
												np.add(initial_kinematics[:,2],gan_pos_forward_frame[:,2]),
												gan_mom_forward_frame[:,0],
												gan_mom_forward_frame[:,1],
												gan_mom_forward_frame[:,2]],0,1)

				initial_kinematics = final_kinematics

				history = np.append(history,[initial_kinematics],axis=0)


			plot_particles = 10

			plt.figure(figsize=(8,8))

			plt.subplot(2,2,1)
			plt.plot(history[:,:plot_particles,2],history[:,:plot_particles,0])
			plt.xlabel('Z Coordinate (cm)')
			plt.ylabel('X Coordinate (cm)')

			plt.subplot(2,2,2)
			total_mom = np.sqrt(np.add(np.add(history[:,:,3]**2,history[:,:,4]**2),history[:,:,5]**2))
			plt.plot(history[:,:plot_particles,2],total_mom[:,:plot_particles])

			plt.xlabel('Z Coordinate (cm)')
			plt.ylabel('Total Momentum (GeV)')

			plt.subplot(2,2,3)
			plt.plot(history[:,:plot_particles,2],history[:,:plot_particles,1])
			plt.xlabel('Z Coordinate (cm)')
			plt.ylabel('Y Coordinate (cm)')

			plt.subplot(2,2,4)
			hist = np.histogram2d(history[-1,:number_of_muons_sim,0],history[-1,:number_of_muons_sim,1],bins=1,range=[[-0.25,0.25],[-0.25,0.25]])
			plt.text(-0.2,0.2,'%d/%d'%(np.sum(hist[0]),number_of_muons_sim))
			plt.hist2d(history[-1,:number_of_muons_sim,0],history[-1,:number_of_muons_sim,1],bins=100,norm=LogNorm(),range=[[-0.25,0.25],[-0.25,0.25]])
			plt.xlabel('X Coordinate (cm)')
			plt.ylabel('Y Coordinate (cm)')
			plt.colorbar()


			plt.savefig('%sprop/prop_1_75_%d.png'%(output_location,e), bbox_inches='tight')
			plt.savefig('%sprop_1_75.png'%output_location, bbox_inches='tight')

			return

		def propagate_z_distance_1_75_single(number_of_muons,GAN_steps):

			def preprocess_pz(input_array, minimum, full_range):
				input_array += minimum * (-1)
				input_array = input_array/full_range

				return input_array

			def unpreprocess_pz(input_array, minimum, full_range):
				# input_array += minimum * (-1)
				# input_array = input_array/full_range

				input_array = input_array*full_range - minimum

				return input_array

			def propagate_post_process(gan_array):

				def propagate_post_process_inner_un_sqrt(input_array, power):

					signs = np.sign(input_array)

					input_array = np.absolute(input_array)

					input_array = np.power(input_array, 1/power)

					input_array = np.multiply(input_array, signs)

					return input_array

				gan_array[:,1] = propagate_post_process_inner_un_sqrt(gan_array[:,1], 0.3)
				gan_array[:,2] = propagate_post_process_inner_un_sqrt(gan_array[:,2], 0.3)


				gan_array = (gan_array/2) + 0.5

				def propagate_post_process_inner(input_array, minimum, full_range):

					input_array = input_array * full_range
					input_array += minimum

					return input_array

				for array in [gan_array]:
					array[:,0] = propagate_post_process_inner(array[:,0], 0, 400)
					array[:,1] = propagate_post_process_inner(array[:,1], -1.757/2, 1.757)
					array[:,2] = propagate_post_process_inner(array[:,2], -1.757/2, 1.757)
					array[:,3] = propagate_post_process_inner(array[:,3], 0, 1.757)
					array[:,4] = propagate_post_process_inner(array[:,4], -0.25, 0.5)
					array[:,5] = propagate_post_process_inner(array[:,5], -0.25, 0.5)
					array[:,6] = propagate_post_process_inner(array[:,6], 0, 400)

				return gan_array



			inital_x = np.zeros(number_of_muons)
			inital_y = np.zeros(number_of_muons)
			inital_z = np.zeros(number_of_muons)

			inital_px = np.zeros(number_of_muons)
			inital_py = np.zeros(number_of_muons)
			inital_pz = np.random.uniform(0,400,1)*np.ones(number_of_muons)


			initial_kinematics = np.swapaxes([inital_x,inital_y,inital_z,inital_px,inital_py,inital_pz],0,1)

			history = np.empty((0,number_of_muons,6))

			history = np.append(history,[initial_kinematics],axis=0)

			for i in range(0, GAN_steps):

				mom_in = np.sqrt(np.add(np.add(initial_kinematics[:,3]**2,initial_kinematics[:,4]**2),initial_kinematics[:,5]**2))
				
				pz_input = preprocess_pz(mom_in,0,400)

				pz_input = (np.subtract(pz_input,0.5)) * 2

				pz_input = np.expand_dims(pz_input,1)
				pz_input = np.expand_dims(pz_input,1)


				random_dimension = np.random.uniform(-1,1,(number_of_muons,10))
				random_dimension = np.expand_dims(random_dimension,1)

				# print(np.shape(pz_input))

				gan_output = np.squeeze(generator.predict([random_dimension,pz_input]))



				gan_output = propagate_post_process(gan_output)


				gan_output[:,3] = np.ones(np.shape(gan_output[:,3]))*1.75
				# print('gan_output',gan_output[0])


				theta_x = np.arctan(np.divide(-initial_kinematics[:,3],initial_kinematics[:,5]))

				mom_after_rotation_of_theta_x = np.swapaxes([initial_kinematics[:,3]*np.cos(theta_x)+initial_kinematics[:,5]*np.sin(theta_x),initial_kinematics[:,4],-initial_kinematics[:,3]*np.sin(theta_x)+initial_kinematics[:,5]*np.cos(theta_x)],0,1)

				theta_y = np.arctan(np.divide(mom_after_rotation_of_theta_x[:,1],mom_after_rotation_of_theta_x[:,2]))

				# mom_after_rotation_of_theta_x_and_theta_y = np.swapaxes([mom_after_rotation_of_theta_x[:,0],mom_after_rotation_of_theta_x[:,1]*np.cos(theta_y)-mom_after_rotation_of_theta_x[:,2]*np.sin(theta_y),mom_after_rotation_of_theta_x[:,1]*np.sin(theta_y)+mom_after_rotation_of_theta_x[:,2]*np.cos(theta_y)],0,1)



				gan_mom = np.swapaxes([gan_output[:,4],gan_output[:,5],gan_output[:,6]],0,1)
				gan_pos = np.swapaxes([gan_output[:,1],gan_output[:,2],gan_output[:,3]],0,1)

				gan_mom_rotate_back_theta_y = np.swapaxes([gan_mom[:,0],gan_mom[:,1]*np.cos(-theta_y)-gan_mom[:,2]*np.sin(-theta_y),gan_mom[:,1]*np.sin(-theta_y)+gan_mom[:,2]*np.cos(-theta_y)],0,1)
				gan_mom_forward_frame = np.swapaxes([gan_mom_rotate_back_theta_y[:,0]*np.cos(-theta_x)+gan_mom_rotate_back_theta_y[:,2]*np.sin(-theta_x),gan_mom_rotate_back_theta_y[:,1],-gan_mom_rotate_back_theta_y[:,0]*np.sin(-theta_x)+gan_mom_rotate_back_theta_y[:,2]*np.cos(-theta_x)],0,1)

				gan_pos_rotate_back_theta_y = np.swapaxes([gan_pos[:,0],gan_pos[:,1]*np.cos(-theta_y)-gan_pos[:,2]*np.sin(-theta_y),gan_pos[:,1]*np.sin(-theta_y)+gan_pos[:,2]*np.cos(-theta_y)],0,1)
				gan_pos_forward_frame = np.swapaxes([gan_pos_rotate_back_theta_y[:,0]*np.cos(-theta_x)+gan_pos_rotate_back_theta_y[:,2]*np.sin(-theta_x),gan_pos_rotate_back_theta_y[:,1],-gan_pos_rotate_back_theta_y[:,0]*np.sin(-theta_x)+gan_pos_rotate_back_theta_y[:,2]*np.cos(-theta_x)],0,1)





				final_kinematics = np.swapaxes([np.add(initial_kinematics[:,0],gan_pos_forward_frame[:,0]),
											np.add(initial_kinematics[:,1],gan_pos_forward_frame[:,1]),
												np.add(initial_kinematics[:,2],gan_pos_forward_frame[:,2]),
												gan_mom_forward_frame[:,0],
												gan_mom_forward_frame[:,1],
												gan_mom_forward_frame[:,2]],0,1)

				initial_kinematics = final_kinematics

				history = np.append(history,[initial_kinematics],axis=0)


			plot_particles = 10

			plt.figure(figsize=(8,8))

			plt.subplot(2,2,1)
			plt.plot(history[:,:plot_particles,2],history[:,:plot_particles,0])
			plt.xlabel('Z Coordinate (cm)')
			plt.ylabel('X Coordinate (cm)')

			plt.subplot(2,2,2)
			total_mom = np.sqrt(np.add(np.add(history[:,:,3]**2,history[:,:,4]**2),history[:,:,5]**2))
			plt.plot(history[:,:plot_particles,2],total_mom[:,:plot_particles])

			plt.xlabel('Z Coordinate (cm)')
			plt.ylabel('Total Momentum (GeV)')

			plt.subplot(2,2,3)
			plt.plot(history[:,:plot_particles,2],history[:,:plot_particles,1])
			plt.xlabel('Z Coordinate (cm)')
			plt.ylabel('Y Coordinate (cm)')

			plt.subplot(2,2,4)
			hist = np.histogram2d(history[-1,:number_of_muons_sim,0],history[-1,:number_of_muons_sim,1],bins=1,range=[[-0.25,0.25],[-0.25,0.25]])
			plt.text(-0.2,0.2,'%d/%d'%(np.sum(hist[0]),number_of_muons_sim))
			plt.hist2d(history[-1,:number_of_muons_sim,0],history[-1,:number_of_muons_sim,1],bins=100,norm=LogNorm(),range=[[-0.25,0.25],[-0.25,0.25]])
			plt.xlabel('X Coordinate (cm)')
			plt.ylabel('Y Coordinate (cm)')
			plt.colorbar()


			plt.savefig('%sprop/prop_1_75_single_%d.png'%(output_location,e), bbox_inches='tight')
			plt.savefig('%sprop_1_75_single.png'%output_location, bbox_inches='tight')

			return

		number_of_muons_sim = 2500
		GAN_steps_sim = 10

		propagate_z_distance(number_of_muons_sim, GAN_steps_sim)

		propagate_z_distance_1_75(number_of_muons_sim, GAN_steps_sim)

		propagate_z_distance_1_75_single(number_of_muons_sim, GAN_steps_sim)


		generator.save('%sGenerator.h5'%output_location)


		print('Saving complete...')


