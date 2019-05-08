from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Input, Dense, Dropout, BatchNormalization, Lambda, GRU, LSTM, Bidirectional, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import Adam, SGD, RMSprop, Nadam, Adagrad, Adadelta

import sys
import numpy as np
import random
import os
import glob
import matplotlib.pyplot as plt

class MetaAgent():
	def __init__(self,use_saved):
		# Input shape
		self.latent_dim = 100 # latent dimension
		self.state_dim = 561 # state dimension
		self.action_dim = 20 # action dimension
		self.deg_pack = 16 # packing degree (number of samples to send to discriminator)
		self.forward_dim = 1154

		#Generator
		g_opt = Adam(0.0002, 0.5, clipnorm=1)
		# opt = SGD(0.001)
		# opt = RMSprop(lr=0.001)
		# opt = Nadam(0.001, 0.5, clipnorm=1)
		# g_opt = Adagrad(lr=0.01, epsilon=None, decay=0.0, clipnorm = 5)
		#opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0, clipnorm = 3)

		#Discriminator
		d_opt = SGD(0.001)

		# build generator and discriminator
		if use_saved:
			self.generator = load_model('{}/generator.h5'.format(use_saved))
			self.discriminator = load_model('{}/discriminator.h5'.format(use_saved))
		else:
			self.generator = self.build_generator()
			self.generator.compile(loss=['categorical_crossentropy'], optimizer = g_opt)
			self.discriminator = self.build_discriminator()
			self.discriminator.compile(loss=['binary_crossentropy'],
					optimizer=d_opt,
					metrics=['accuracy'])

		# the generator takes noise and states as input
		# and returns actions for each state
		noise = Input(shape=(self.latent_dim,))
		states = Input(shape=(self.deg_pack, self.state_dim))

		unstacked_states = Lambda(lambda x: K.tf.unstack(x, axis=1))(states)
		unstacked_actions = [self.generator([noise, state]) for state in unstacked_states]
		actions = Lambda(lambda x: K.stack(x, axis=1))(unstacked_actions)
		
		# don't train the discriminator while training the generator
		self.discriminator.trainable = False

		# the discriminator takes generated actions and game states
		# and determines validity
		valid = self.discriminator([states, actions])

		self.combined = Model([noise, states], valid)
		self.combined.compile(loss=['binary_crossentropy'], optimizer=g_opt)

	def build_generator(self):
		model = Sequential([
			Dense(256, input_dim=self.latent_dim + self.state_dim),
			LeakyReLU(alpha=0.2),
			BatchNormalization(momentum=0.8),
			Dense(128),
			LeakyReLU(alpha=0.2),
			BatchNormalization(momentum=0.8),
			Dense(self.action_dim, activation='sigmoid'),
		])

		noise = Input(shape=(self.latent_dim,))
		state = Input(shape=(self.state_dim,), dtype='float32')
		model_input = concatenate([noise, state], 1)

		return Model([noise, state], model(model_input))

	def build_discriminator(self):
		model = Sequential([
			Flatten(),
			# Bidirectional(LSTM(64,input_shape=(self.state_dim+self.action_dim,))),
			Dense(300, kernel_regularizer=l2(0.001)),
			LeakyReLU(alpha=0.2),
			Dropout(0.5),
			Dense(200, kernel_regularizer=l2(0.001)),
			LeakyReLU(alpha=0.2),
			Dropout(0.5),
			Dense(100, kernel_regularizer=l2(0.001)),
			LeakyReLU(alpha=0.2),
			Dropout(0.5),
			Dense(1, activation='sigmoid'),
		])

		states = Input(shape=(self.deg_pack, self.state_dim))
		actions = Input(shape=(self.deg_pack, self.action_dim))
		state_actions = concatenate([states, actions],2)

		return Model([states,actions],model(state_actions))

	def load_data(self):
		print("loading data")
		# ds is a numpy array of shape (num_agents, num_turns, state_dim + action_dim)
		ds = []
		for agent in glob.glob("data/*.txt"):
			d = []
			with open(agent, "rb") as f:
				# while True:
				for i in range(40000):
					turn = f.read(self.state_dim + self.action_dim + self.forward_dim)
					if not turn:
						break
					if turn[0] == 45:
						# break
						continue
					d.append([bit-48 for bit in turn[:581]])
			print("done loading agent {}".format(agent))
			ds.append(d)
		self.ds = ds
		# check if all values are either 1 or 0
		# print("data is good:", np.all([ b==0 or b==1 for b in np.concatenate([np.array(agent) for agent in ds], axis=0).flatten() ]))
		print("done loading data")

	def get_samples(self, num_samples):
		s = np.array([random.sample(agent, self.deg_pack) for agent in random.choices(self.ds, k=num_samples) ])
		return s[:,:,:self.state_dim], s[:,:,self.state_dim:]

	def train(self, epochs, batch_size=128, sample_interval=50):
		
		# adversarial ground truths
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))
		noisy_valid = np.random.uniform(0.7, 1.0, (batch_size, 1))
		noisy_fake = np.random.uniform(0, 0.3, (batch_size, 1))

		try:
			os.mkdir('model')
		except:
			pass
		# Array initialization for logging of the losses
		d_loss_logs_r = []
		d_loss_logs_f = []
		g_loss_logs = []

		try:
			for epoch in range(epochs):

				# ---------------------
				#  Train Discriminator
				# ---------------------

				# state_samples is a tensor of shape (batch_size, deg_pack, state_dim)
				# action_samples is a tensor of shape (batch_size, deg_pack, action_dim)
				state_samples, action_samples = self.get_samples(batch_size)

				noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
				repeated_noise = noise.repeat(self.deg_pack, axis=0)

				flattened = np.reshape(state_samples, (batch_size * self.deg_pack, -1))
				gen_action_samples = self.generator.predict([repeated_noise, flattened]).reshape((batch_size, self.deg_pack, -1))

				d_loss_real = self.discriminator.train_on_batch([state_samples, action_samples], valid)
				d_loss_fake = self.discriminator.train_on_batch([state_samples, gen_action_samples], fake)
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

				# ---------------------
				#  Train Generator
				# ---------------------

				state_samples, _ = self.get_samples(batch_size)

				g_loss = self.combined.train_on_batch([noise, state_samples], valid)

				print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

				if np.isnan(d_loss[0]) or np.isnan(g_loss):
					break

				# Store the losses
				d_loss_logs_r.append([epoch, d_loss[0]])
				d_loss_logs_f.append([epoch, d_loss[1]])
				g_loss_logs.append([epoch, g_loss])

				if epoch % sample_interval == 0:
					self.generator.save('model/generator.h5')
					self.discriminator.save('model/discriminator.h5')

		except KeyboardInterrupt:
			pass

		d_loss_logs_r_a = np.array(d_loss_logs_r)
		d_loss_logs_f_a = np.array(d_loss_logs_f)
		g_loss_logs_a = np.array(g_loss_logs)

		# At the end of training plot the losses vs epochs
		plt.plot(d_loss_logs_r_a[:,0], d_loss_logs_r_a[:,1], label="Discriminator Loss - Real")
		plt.plot(d_loss_logs_f_a[:,0], d_loss_logs_f_a[:,1], label="Discriminator Loss - Fake")
		plt.plot(g_loss_logs_a[:,0], g_loss_logs_a[:,1], label="Generator Loss")
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.title('GAN')
		plt.grid(True)
		plt.show()

if __name__ == '__main__':
	use_saved = sys.argv[1] if len(sys.argv) > 1 else None
	meta_agent = MetaAgent(use_saved)
	meta_agent.load_data()
	meta_agent.train(epochs=1000, batch_size=256, sample_interval=200)
