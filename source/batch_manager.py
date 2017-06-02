#This class holds our data and releases batches potentially with random transformations

from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
class batch_manager(object): 
	"""The batch manager class manages the batches. It is responsible for preprocessing and returning batches."""
	def __init__(self, data, labels, valid_num, OneHot = True, aug_params = None, shape = [32, 32, 3]): 
		"""valid_num : number of samples to set aside for testing. Taken from the end of data and labels.
		OneHot : if True one hot encode the labels
		aug_params : a dictionary of data preprocessing commands to be past to the keras image preprocessor.
		shape : the shape of the image samples 
		"""
		self.shape = shape

		#cut the data 
		self.train_data = data[0:-valid_num].reshape([-1]+shape)
		self.valid_data = data[-valid_num:].reshape([-1]+shape)

		#One hot encode if necessary 
		if OneHot: 
			self.enc = OneHotEncoder(sparse = False) 
			labels = self.enc.fit_transform(labels.reshape(-1,1)) 

		#cut the labels 
		self.train_labels = labels[0:-valid_num] 
		self.valid_labels = labels[-valid_num:] 

		#current epoch
		self.epoch = 1 
		#number of samples we have looked in current epoch
		self.batched = 0 
		#total number of samples
		self.total_batched = 0

		#number of entries 
		self.train_length = self.train_labels.shape[0] 
		self.valid_length = valid_num 
			    	
		#number of labels should match data length 
		assert(self.train_length == len(self.train_data)) 

		#Whether or not to distort the data
		self.distort = (aug_params != None) 
		if self.distort: 
			#Build a moderate data augmentor using parameters 
			self.idg = tf.contrib.keras.preprocessing.image.ImageDataGenerator(**aug_params) 
			#set transformation (note that for some reason, on grey scale images the output of the transform
			#is squeezed yielding an inconsistent shape
			self.trans = lambda x: self.idg.random_transform(x)

	# def get_batch(self, batch_size, distort = False, shuffle = True): 
	# 	"""Returns the next batch for training.
	# 	    distort : if true apply data augmentation preprocessing.
	# 	    shuffle : if true return a random selection of samples
	# 	"""
	# 	if self.batched+batch_size >= self.train_length: 
	# 		#increase epoch 
	# 		self.epoch+=1
		
	# 	#Random batch
	# 	if shuffle: 
	# 		selections = np.random.randint(self.train_length, size = batch_size) 
	# 	#Sequential batch 
	# 	else:
	# 		#if we are about to complete an epoch wrap the batch
	# 		if self.batched+batch_size >= self.train_length:
	# 			selections = list(range(batch_size - (self.train_length - self.batched))) + list(range(self.batched, self.train_length))
	# 		else: 
	# 			selections = range(self.batched, self.batched + batch_size)
	# 	data, labels = self.train_data[selections, :], self.train_labels[selections, :] 


	# 	#Increment our counters 
	# 	self.total_batched += batch_size 
	# 	self.batched = (batch_size+self.batched) % self.train_length 

	# 	if distort:
 #        			#in place transformation
 #            		for i, im in enumerate(data):
	# 	                data[i]=self.trans(im)
	# 	return data, labels 

	def train_batches(self, batch_size, shuffle = False, distort = False):
 		"""This is a generator for yielding validation batches. Batch validation is required when we have limited memory."""
 		batched = 0 

 		while batched <= self.train_length: 
 			if shuffle: 
 				selections = np.random.randint(self.train_length, size = batch_size) 
 				data, labels = self.train_data[selections], self.train_labels[selections] 
 			else: 
 				data, labels = self.train_data[batched: batched + batch_size], self.train_labels[batched : batched + batch_size] 
 			if distort: 
 				#in place transformation 
 				for i, im in enumerate(data): 
 					data[i]=self.trans(im) 
 			batched+=batch_size 
 			yield data, labels
 		# #last, potentially shorter batch 
 		# yield self.train_data[batched : ], self.train_labels[batched : ]

	def valid_batches(self, batch_size): 
		"""This is a generator for yielding validation batches. Batch validation is required when we have limited memory."""
		batched = 0

		while batched + batch_size < self.valid_length:
			yield self.valid_data[batched: batched + batch_size], self.valid_labels[batched : batched + batch_size]
			batched+=batch_size
		#last, potentially shorter batch
		yield self.valid_data[batched : ], self.valid_labels[batched : ]

	# def batch_gen(data, labels, batch_size):
	# 	batched = 0
	# 	while batched + batch_size < len(data):
	# 		yield data[batched: batched + batch_size], labels[batched : batched + batch_size]
	# 		batched+=batch_size
	# 	#last, potentially shorter batch
	# 	yield data[batched : ], labels[batched : ]		
