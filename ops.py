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




def propagate_post_process_2(gan_array):

	def propagate_post_process_inner_un_sqrt(input_array, power):

		signs = np.sign(input_array)

		input_array = np.absolute(input_array)

		input_array = np.power(input_array, 1/power)

		input_array = np.multiply(input_array, signs)

		return input_array

	gan_array[:,:,1] = propagate_post_process_inner_un_sqrt(gan_array[:,:,1], 0.3)
	gan_array[:,:,2] = propagate_post_process_inner_un_sqrt(gan_array[:,:,2], 0.3)

	gan_array = (gan_array/2) + 0.5

	def propagate_post_process_inner(input_array, minimum, full_range):

		input_array = input_array * full_range
		input_array += minimum

		return input_array

	for array in [gan_array]:
		array[:,:,0] = propagate_post_process_inner(array[:,:,0], 0, 400)
		array[:,:,1] = propagate_post_process_inner(array[:,:,1], -1.757/2, 1.757)
		array[:,:,2] = propagate_post_process_inner(array[:,:,2], -1.757/2, 1.757)
		array[:,:,3] = propagate_post_process_inner(array[:,:,3], 0, 1.757)
		array[:,:,4] = propagate_post_process_inner(array[:,:,4], -0.25, 0.5)
		array[:,:,5] = propagate_post_process_inner(array[:,:,5], -0.25, 0.5)
		array[:,:,6] = propagate_post_process_inner(array[:,:,6], 0, 400)

	return gan_array




def propagate_post_process_1(gan_array):

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


def preprocess_pz(input_array, minimum, full_range):
	input_array += minimum * (-1)
	input_array = input_array/full_range

	return input_array


def plot_1d_hists(dim, sample_to_test, synthetic_test_output, output_location, e):

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

def propagate_z_distance(number_of_muons,GAN_steps, mode, output_location, e, generator, number_of_noise_dims):

	'''
	mode:

		0 - full GAN
		1 - GAN with fixed Z
		2 - single muon

	'''

	inital_x = np.zeros(number_of_muons)
	inital_y = np.zeros(number_of_muons)
	inital_z = np.zeros(number_of_muons)

	inital_px = np.zeros(number_of_muons)
	inital_py = np.zeros(number_of_muons)

	if mode == 0: inital_pz = np.random.uniform(0,400,number_of_muons)
	if mode == 1: inital_pz = np.random.uniform(0,400,number_of_muons)
	if mode == 2: inital_pz = np.random.uniform(0,400,1)*np.ones(number_of_muons)


	initial_kinematics = np.swapaxes([inital_x,inital_y,inital_z,inital_px,inital_py,inital_pz],0,1)

	history = np.empty((0,number_of_muons,6))

	history = np.append(history,[initial_kinematics],axis=0)

	for i in range(0, GAN_steps):

		mom_in = np.sqrt(np.add(np.add(initial_kinematics[:,3]**2,initial_kinematics[:,4]**2),initial_kinematics[:,5]**2))
		
		pz_input = preprocess_pz(mom_in,0,400)

		pz_input = (np.subtract(pz_input,0.5)) * 2

		pz_input = np.expand_dims(pz_input,1)
		pz_input = np.expand_dims(pz_input,1)


		random_dimension = np.random.uniform(-1,1,(number_of_muons,number_of_noise_dims))
		random_dimension = np.expand_dims(random_dimension,1)

		# print(np.shape(pz_input))

		gan_output = np.squeeze(generator.predict([random_dimension,pz_input]))



		gan_output = propagate_post_process_1(gan_output)

		if mode == 1: gan_output[:,3] = np.ones(np.shape(gan_output[:,3]))*1.75
		if mode == 2: gan_output[:,3] = np.ones(np.shape(gan_output[:,3]))*1.75


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
	hist = np.histogram2d(history[-1,:number_of_muons,0],history[-1,:number_of_muons,1],bins=1,range=[[-0.25,0.25],[-0.25,0.25]])
	plt.text(-0.2,0.2,'%d/%d'%(np.sum(hist[0]),number_of_muons))
	plt.hist2d(history[-1,:number_of_muons,0],history[-1,:number_of_muons,1],bins=100,norm=LogNorm(),range=[[-0.25,0.25],[-0.25,0.25]])
	plt.xlabel('X Coordinate (cm)')
	plt.ylabel('Y Coordinate (cm)')
	plt.colorbar()

	if mode == 0:
		plt.savefig('%sprop/prop_%d.png'%(output_location,e), bbox_inches='tight')
		plt.savefig('%sprop.png'%output_location, bbox_inches='tight')
		plt.close('all')
	elif mode == 1:
		plt.savefig('%sprop/prop_1_75_%d.png'%(output_location,e), bbox_inches='tight')
		plt.savefig('%sprop_1_75.png'%output_location, bbox_inches='tight')
		plt.close('all')
	elif mode == 2:
		plt.savefig('%sprop/prop_1_75_single_%d.png'%(output_location,e), bbox_inches='tight')
		plt.savefig('%sprop_1_75_single.png'%output_location, bbox_inches='tight')
		plt.close('all')

	return


def plot_bethe_bloch(points,points_per_point, output_location, generator, number_of_noise_dims):

	start = 1
	end = 400
	points_to_test = np.arange(start,end,(end-start)/points)
	mean_at_points = np.empty(0)

	# print(points_to_test)

	for x in points_to_test:
		# print(x)
		# print(points_to_test, np.shape(points_to_test))

		random_dimension = np.random.uniform(-1,1,(points_per_point,number_of_noise_dims))
		random_dimension = np.expand_dims(random_dimension,1)

		points_to_test_in = np.ones(points_per_point)*x

		points_to_test_in = np.expand_dims(points_to_test_in,1)
		points_to_test_in = np.expand_dims(points_to_test_in,1)

		points_to_test_in = preprocess_pz(points_to_test_in,0,400)
		points_to_test_in = (np.subtract(points_to_test_in,0.5)) * 2

		gan_output = np.squeeze(generator.predict([random_dimension,points_to_test_in]))

		

		gan_output = propagate_post_process_1(gan_output)

		mom_in = gan_output[:,0]

		mom_out = np.sqrt(np.add(np.add(gan_output[:,4]**2,gan_output[:,5]**2),gan_output[:,6]**2))

		mean_at_points = np.append(mean_at_points, np.mean(np.subtract(mom_in,mom_out)))


	plt.plot(points_to_test, mean_at_points)
	plt.ylabel('Mean momentum change over GAN (GeV)')
	plt.xlabel('Input momentum (GeV)')
	plt.savefig('%sbethe_bloch.png'%output_location, bbox_inches='tight')
	plt.close('all')

	return