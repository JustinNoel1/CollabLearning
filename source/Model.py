from resnet import *
from batch_manager import *
from Logger import *
import tensorflow as tf 
#Generic model holds features common to all models
class Model(object): 
	def __init__(self, data, labels, params): 
		#parameters 
		self.img_shape = [None] + params['image_shape'] 
		self.params = params
		self.batch_size = params['batch_size'] 

		#Placeholders 
		self.x =  tf.placeholder( tf.float32, shape = self.img_shape, name = "x") 
		self.y =  tf.placeholder( tf.float32, shape =(None, 10), name = "y") 
		self.training = tf.placeholder( tf.bool, shape = ()) 

		#if we are truncating the data (for debugging purposes), make the dataset roughly 500x smaller 
		if params['truncate_data']: 
			self.batch = batch_manager(data[:min(len(data)//50, 500)], labels[:min(len(labels)//50, 500)], 
				valid_num = min(params['valid_num']//500, 50), aug_params = params['data_augmentation'], 
				shape = params['image_shape'])
		else: 
			self.batch = batch_manager(data, labels, valid_num = params['valid_num'], 
				aug_params = params['data_augmentation'], shape = params['image_shape']) 

		#more parameters
		self.batch_size = params['batch_size'] 
		self.epochs = params['epochs'] 
		self.train_size = self.batch.train_length 
		self.distort = (params['data_augmentation'] != None) 
		self.shuffle = params['shuffle']

	def train(self, sess): 
		# Initializing the variables 
		sess.run(tf.global_variables_initializer()) 

		previous_epoch = self.batch.epoch 

		#initialize feed_dict 
		feed_dict = {} 

		counter = 0 
		while self.batch.epoch <= self.epochs:
	    		#load feed_dict 
	    		# feed_dict[self.x], feed_dict[self.y] = self.batch.get_batch(self.batch_size, shuffle = self.shuffle, 
	    		# 	distort = self.distort) 
	    		# feed_dict[self.training]  = True
	    		log_data = self.optimize(sess, feed_dict, self.batch.epoch)
	    		self.log_batch(log_data) 

	    		#End of epoch testing
	    		# if previous_epoch != self.batch.epoch: 
	    		# 	previous_epoch = self.batch.epoch 

    			log_data = self.get_errors(sess, feed_dict, self.batch.epoch)
    			self.log_epoch(log_data, sess)

	#Interface that must be implemented by instances
	def optimize(self, sess, feed_dict, epoch):
		pass

	def get_errors(self, sess, feed_dict, epoch):
		pass

	def log_batch(self, log_data):
		pass

	def log_epoch(self, log_data, sess):
		pass

	#learning_rate schedule
	def get_learning_rate2(self, epoch): 
		lr = self.params['learning_rate']
		if epoch > 0.9 * self.epochs: 
			return lr*(0.5**3) 
		elif epoch >= 0.75*self.epochs: 
			return lr*(0.5**2) 
		elif epoch >= 0.5*self.epochs: 
			return lr * 0.5 
		else: 
			return lr
	def get_learning_rate(self, epoch): 
		lr = self.params['learning_rate']
		if epoch > 0.5 * self.epochs: 
			return lr*(0.1) 
		# elif epoch >= 0.75*self.epochs: 
		# 	return lr*(0.5**2) 
		# elif epoch >= 0.5*self.epochs: 
		# 	return lr * 0.5 
		else: 
			return lr
	@staticmethod
	def get_model(data, labels, params): 
		"""Returns the desired model object from the parameters."""
		if params['model_type'] == 'single': 
			return SingleModel(data, labels, params) 
		else:
			return EnsembleModelNew(data, labels, params)

class SingleModel(Model):
	def __init__(self, data, labels, params): 
		from math import ceil 
		Model.__init__(self, data, labels, params)
		self.regularization_rate = params['regularization_rate'] 

		cur_scope = params['filename']
		with tf.name_scope(cur_scope): 
			self.learning_rate = tf.placeholder(tf.float32, shape = []) 
			#logits of our model 
			self.logits = deep_model(self.x, self.training, params = params, scope = cur_scope) 

			#Cost function is cross entropy plus L2 weight decay 
			self.cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.logits), name = "cross_ent") 

			#Don't apply weight decay to bias variables (beta is the name in batch_normalization layers)
			if self.regularization_rate == 0.:
				self.cost = self.cross_ent
			else:
				self.regularizer = tf.contrib.layers.l2_regularizer(scale = self.regularization_rate )
				self.lossL2 =tf.reduce_sum( [self.regularizer(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = cur_scope) if ('bias' not in v.name) and ('beta' not in v.name)])
				self.cost = self.cross_ent + self.lossL2

			#predictions 
			self.preds = tf.argmax(self.logits, 1, name = 'preds') 

			#Accuracy statistics 
			self.correct_pred = tf.equal(self.preds, tf.argmax(self.y, 1), name = 'correct_pred') 
			self.incorrect_pred = tf.not_equal(self.preds, tf.argmax(self.y, 1), name = 'correct_pred') 
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy') 
			self.error = 1.0-self.accuracy 

			#Apparently we need this next line to update batch normalization parameters 
			self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 

			#Optimization step 
			with tf.control_dependencies(self.extra_update_ops): 
				self.ao = tf.train.AdamOptimizer(self.learning_rate) 
				self.optimizer = self.ao.minimize(self.cost)

			#Logger must be created after we have built the graph	
			self.logger = Logger(params, ceil(self.batch.train_length/self.batch_size)) 


	def optimize(self, sess, feed_dict, epoch): 
		feed_dict[self.training]  = True 
		feed_dict[self.learning_rate] = self.get_learning_rate(epoch)
		cost = 0.
		error = 0.

		#Go through the validation set in batches (to avoid memory overruns). 
		#Sum up the unaveraged error statistics
		for feed_dict[self.x], feed_dict[self.y] in self.batch.train_batches(self.batch_size, shuffle = self.shuffle,
			distort = self.distort):
			_, c, e = sess.run([self.optimizer, self.cost, self.error], feed_dict = feed_dict)
			# From previous version
			# cost += 0.95*cost + 0.05*c
			# error = 0.95*error + 0.05*e
			cost += c*len(feed_dict[self.y])
			error += e*len(feed_dict[self.y])

		self.batch.epoch+=1

		return {'train_cost' : cost/self.batch.train_length, 'train_error' : error/self.batch.train_length }

	def get_errors(self, sess, feed_dict, epoch): 
		feed_dict[self.training]  = False 
		feed_dict[self.learning_rate] = self.get_learning_rate(epoch)
		cost = 0.
		error = 0.

		#Go through the validation set in batches (to avoid memory overruns). 
		#Sum up the unaveraged error statistics
		for feed_dict[self.x], feed_dict[self.y] in self.batch.valid_batches(self.batch_size):
			c, e = sess.run([self.cost, self.error], feed_dict = feed_dict)
			cost += c*len(feed_dict[self.y])
			error += e*len(feed_dict[self.y])

		return {'test_cost' : cost / self.batch.valid_length, 'test_error' : error/self.batch.valid_length}

	def log_batch(self, log_data):
		self.logger.batch_update(log_data['train_error'], log_data['train_cost'])

	def log_epoch(self, log_data, sess):
		"""This function logs the validation data after an epoch. It will be overridden by more complicated models."""
		self.epoch_logger(log_data, sess, self.logger)

	def epoch_logger(self, log_data, sess, logger):
		"""Reusable method for ensemble learners."""
		logger.epoch_update(log_data['test_error'], log_data['test_cost'], sess)

class EnsembleModelNew(Model):
	def __init__(self, data, labels, params): 
		from math import ceil 
		Model.__init__(self, data, labels, params)
		self.list_of_params = params['list_of_params']
		self.ensemble_size = len(self.list_of_params)
		self.learning_rate = tf.placeholder(tf.float32, shape = []) 
		#Convenience  variable
		ensemble_size = self.ensemble_size

		self.base_name  = params['filename']
		self.extra_ops = []

		#initialize arrays
		self.regularization_rate = [None]*ensemble_size
		self.regularizer = [None]*ensemble_size
		# self.learning_rate = [None]*ensemble_size
		self.logits = [None]*ensemble_size
		self.cross_ent = [None]*ensemble_size
		self.tvars = [None]*ensemble_size
		self.lossL2 = [None]*ensemble_size
		self.cost = [None]*ensemble_size
		self.correct_pred = [None]*ensemble_size
		self.preds = [None]*ensemble_size
		self.accuracy = [None]*ensemble_size
		self.error = [None]*ensemble_size
		self.extra_update_ops = [None]*ensemble_size
		self.logger = [None]*ensemble_size
		self.scores = [None]*ensemble_size
		self.incorrect_pred = [None]*ensemble_size
		self.optimizers = [None]*ensemble_size
		self.aos = [None]*ensemble_size

		for i in range(self.ensemble_size):
			self.regularization_rate[i] = self.list_of_params[i]['regularization_rate']

			cur_scope = self.base_name + "_{0:03d}".format(i)
			with tf.name_scope( cur_scope ): 
				#logits of our model 
				self.logits[i] = deep_model(self.x, self.training, params = self.list_of_params[i], scope = cur_scope ) 

				#Cost function is cross entropy plus L2 weight decay 
				self.cross_ent[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, 
					logits = self.logits[i]), name = "cross_ent")

				#Don't apply weight decay to bias variables (beta is the name in batch_normalization layers)
				if self.regularization_rate[i]==0.:
					self.cost[i] = self.cross_ent[i]
				else:
					self.regularizer[i] = tf.contrib.layers.l2_regularizer(scale = self.regularization_rate[i]) 
					self.lossL2[i] = tf.reduce_sum( [self.regularizer[i](v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = cur_scope) if ('bias' not in v.name) and ('beta' not in v.name)]) 
					self.cost[i] = self.cross_ent[i] + self.lossL2[i]

				#predictions 
				self.scores[i] = tf.nn.softmax(self.logits[i], name = 'scores') 
				self.preds[i] = tf.argmax(self.logits[i], 1, name = 'preds') 

				#Accuracy statistics 
				self.correct_pred[i] = tf.equal(self.preds[i], tf.argmax(self.y, 1), name = 'correct_pred') 
				self.incorrect_pred[i] = tf.not_equal(self.preds[i], tf.argmax(self.y, 1), name = 'incorrect_pred') 
				self.accuracy[i] = tf.reduce_mean(tf.cast(self.correct_pred[i], tf.float32), name='accuracy') 
				self.error[i] = 1.0-self.accuracy[i] 

				#Apparently we need this next line to update batch normalization parameters 
				self.extra_update_ops[i] = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = cur_scope) 
				self.extra_ops += self.extra_update_ops[i]

				#Optimization step 
				with tf.control_dependencies(self.extra_update_ops[i]): 
					self.aos[i] = tf.train.AdamOptimizer(self.learning_rate) 
					self.optimizers[i] = self.aos[i].minimize(self.cost[i])

				#Logger must be created after we have built the graph	
				params['filename']=cur_scope
				self.logger[i] = Logger(params, ceil(self.batch.train_length/self.batch_size)) 

		params['filename'] = self.base_name + "_Ensemble"
		self.ens_logger = Logger(params, ceil(self.batch.train_length/self.batch_size)) 
		params['filename'] = self.base_name
		self.total_cost = tf.reduce_sum(self.cost, axis = 0)
		self.ens_logits = tf.reduce_mean(self.logits, axis = 0)
		self.ens_cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits = self.ens_logits))
		self.ens_preds = tf.argmax(self.ens_logits, 1, name = 'ens_preds') 
		self.ens_correct_pred = tf.equal(self.ens_preds, tf.argmax(self.y, 1), name = 'ens_correct_pred')
		self.ens_incorrect_pred = tf.not_equal(self.ens_preds, tf.argmax(self.y, 1), name = 'ens_incorrect_pred') 
		self.ens_accuracy = tf.reduce_mean(tf.cast(self.ens_correct_pred, tf.float32), name='ensaccuracy') 
		self.ens_error = 1.0-self.ens_accuracy

		if params['model_type'] == 'collab':
			#get neighboring cross entropies
			self.ces = [None]*(self.ensemble_size-1)
			for i in range(self.ensemble_size-1):
				self.ces[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.scores[i+1], logits = self.logits[i])) 	+ tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.scores[i], logits = self.logits[i+1]))

			self.total_cross_ent = tf.reduce_sum(self.ces)

			#get neighboring sum of square differences in logits
			self.sum_of_squares = [None]*(self.ensemble_size-1)
			for i in range(self.ensemble_size-1):
				self.sum_of_squares[i] = tf.reduce_sum((self.logits[i]-self.logits[i+1])**2, axis =1)

			self.total_sum_of_squares = tf.reduce_mean(self.sum_of_squares)

			#get neighboring sum of absolute differences in logits
			self.sum_of_dists = [None]*(self.ensemble_size-1)
			for i in range(self.ensemble_size-1):
				self.sum_of_dists[i] = tf.reduce_sum(tf.abs(self.logits[i]-self.logits[i+1]), axis=1)

			self.total_sum_of_dists = tf.reduce_mean(self.sum_of_dists)

			self.collab_weight = params['collab_weight']
			if params['collab_method'] == 'cross_ent':
				self.total_cost += self.total_cross_ent*self.collab_weight
			elif params['collab_method'] ==  'L2':
				self.total_cost += self.total_sum_of_squares*self.collab_weight
			elif params['collab_method'] == 'L1':
				self.total_cost += self.total_sum_of_dists*self.collab_weight

		#Optimization step 
		with tf.control_dependencies(self.extra_ops): 
				self.ao = tf.train.AdamOptimizer(self.learning_rate) 
				self.optimizer = self.ao.minimize(self.total_cost)

	def optimize(self, sess, feed_dict, epoch): 
		feed_dict[self.training]  = True 
		feed_dict[self.learning_rate] = self.get_learning_rate(epoch)
		cost = np.zeros(self.ensemble_size)
		error = np.zeros(self.ensemble_size)
		stats = np.zeros(self.ensemble_size*2+2)
		ens_c = 0.
		ens_e = 0.
		#Go through the validation set in batches (to avoid memory overruns). 
		#Sum up the unaveraged error statistics
		for feed_dict[self.x], feed_dict[self.y] in self.batch.train_batches(self.batch_size, 
			shuffle = self.shuffle, distort = self.distort):
			_,  *stats = sess.run([self.optimizer, *self.cost, *self.error, self.ens_cross_ent, self.ens_error], feed_dict = feed_dict)
			#previous way of measuring stats
			# stats = 0.05*np.array(stats)
			# cost = 0.95*cost + stats[0:self.ensemble_size]
			# error = 0.95*error + stats[self.ensemble_size : 2*self.ensemble_size]
			# ens_c = 0.95*ens_c + stats[2*self.ensemble_size]
			# ens_e = 0.95*ens_e + stats[2*self.ensemble_size+1]
			cost += len(feed_dict[self.y])*stats[0:self.ensemble_size]
			error += len(feed_dict[self.y])*stats[self.ensemble_size : 2*self.ensemble_size]
			ens_c += len(feed_dict[self.y])*stats[2*self.ensemble_size]
			ens_e += len(feed_dict[self.y])*stats[2*self.ensemble_size+1]
		self.batch.epoch+=1


		#wrong_preds += w
		log_data = []
		for i in range(self.ensemble_size):
			log_data.append({'train_cost' : cost[i]/self.batch.train_length, 'train_error' : error[i]/self.batch.train_length})
		log_data.append({'ensemble_train_error' : ens_e/self.batch.train_length, 
			'ensemble_train_cost' : ens_c/self.batch.train_length})

		return log_data

	def get_errors(self, sess, feed_dict, epoch): 
		feed_dict[self.training]  = False 
		feed_dict[self.learning_rate] = self.get_learning_rate(epoch)
		cost = np.zeros(self.ensemble_size)
		error = np.zeros(self.ensemble_size)
		# stats = np.zeros(self.ensemble_size*2+2)
		ens_c = 0.
		ens_e = 0.
		#Go through the validation set in batches (to avoid memory overruns). 
		#Sum up the unaveraged error statistics
		for feed_dict[self.x], feed_dict[self.y] in self.batch.valid_batches(self.batch_size):
			stats = sess.run([*self.cost, *self.error, self.ens_cross_ent, self.ens_error], feed_dict = feed_dict)
			stats = np.array(stats)
			# print(stats[0:self.ensemble_size])
			cost += len(feed_dict[self.y])*stats[0:self.ensemble_size]
			error += len(feed_dict[self.y])*stats[self.ensemble_size : 2*self.ensemble_size]
			ens_c += len(feed_dict[self.y])*stats[2*self.ensemble_size]
			ens_e += len(feed_dict[self.y])*stats[2*self.ensemble_size+1]
		log_data = []
		for i in range(self.ensemble_size):
			log_data.append({'test_cost' : cost[i]/self.batch.valid_length, 'test_error' : error[i]/self.batch.valid_length})
		log_data.append({'ensemble_test_error' : ens_e/self.batch.valid_length, 'ensemble_test_cost' : ens_c/self.batch.valid_length})

		return log_data

	def log_batch(self, log_data):
		for i in range(self.ensemble_size):
			self.logger[i].batch_update(log_data[i]['train_error'], log_data[i]['train_cost'])
		self.ens_logger.batch_update(log_data[-1]['ensemble_train_error'], log_data[-1]['ensemble_train_cost'])

	def log_epoch(self, log_data, sess):
		for i in range(self.ensemble_size):
			self.logger[i].epoch_update(log_data[i]['test_error'], log_data[i]['test_cost'], sess)
		self.ens_logger.epoch_update(log_data[-1]['ensemble_test_error'], log_data[-1]['ensemble_test_cost'], None)


# class EnsembleModel(Model):
# 	def __init__(self, data, labels, params): 
# 		from math import ceil 
# 		Model.__init__(self, data, labels, params)
# 		self.list_of_params = params['list_of_params']
# 		self.ensemble_size = len(self.list_of_params)
# 		#Convenience  variable
# 		ensemble_size = self.ensemble_size

# 		self.base_name  = params['filename']

# 		#initialize arrays
# 		self.learning_rate = [None]*ensemble_size
# 		self.regularization_rate = [None]*ensemble_size
# 		self.logits = [None]*ensemble_size
# 		self.cross_ent = [None]*ensemble_size
# 		self.tvars = [None]*ensemble_size
# 		self.lossL2 = [None]*ensemble_size
# 		self.cost = [None]*ensemble_size
# 		self.correct_pred = [None]*ensemble_size
# 		self.preds = [None]*ensemble_size
# 		self.accuracy = [None]*ensemble_size
# 		self.error = [None]*ensemble_size
# 		self.extra_update_ops = [None]*ensemble_size
# 		self.ao = [None]*ensemble_size
# 		self.optimizer = [None]*ensemble_size
# 		self.logger = [None]*ensemble_size
# 		self.scores = [None]*ensemble_size
# 		self.incorrect_pred = [None]*ensemble_size

# 		for i in range(self.ensemble_size):
# 			self.regularization_rate[i] = self.list_of_params[i]['regularization_rate']

# 			cur_scope = self.base_name + "_{0:03d}".format(i)
# 			with tf.name_scope( cur_scope ): 
# 				self.learning_rate[i] = tf.placeholder(tf.float32, shape = []) 
# 				#logits of our model 
# 				self.logits[i] = deep_model(self.x, self.training, params = self.list_of_params[i], scope = cur_scope ) 

# 				#Cost function is cross entropy plus L2 weight decay 
# 				self.cross_ent[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits = self.logits[i]), 
# 					name = "cross_ent") 

# 				#Don't apply weight decay to bias variables (beta is the name in batch_normalization layers)
# 				self.tvars[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = cur_scope)
# 				self.lossL2[i] = tf.reduce_mean([ tf.nn.l2_loss(v) for v in self.tvars[i] 
# 					if ('bias' not in v.name) and ('beta' not in v.name) ]) 
# 				self.cost[i] = self.cross_ent[i] + self.lossL2[i]*(self.regularization_rate[i]) 

# 				#predictions 
# 				self.scores[i] = tf.nn.softmax(self.logits[i], name = 'scores') 
# 				self.preds[i] = tf.argmax(self.logits[i], 1, name = 'preds') 

# 				#Accuracy statistics 
# 				self.correct_pred[i] = tf.equal(self.preds[i], tf.argmax(self.y, 1), name = 'correct_pred') 
# 				self.incorrect_pred[i] = tf.not_equal(self.preds[i], tf.argmax(self.y, 1), name = 'correct_pred') 
# 				self.accuracy[i] = tf.reduce_mean(tf.cast(self.correct_pred[i], tf.float32), name='accuracy') 
# 				self.error[i] = 1.0-self.accuracy[i] 

# 				#Apparently we need this next line to update batch normalization parameters 
# 				self.extra_update_ops[i] = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = cur_scope) 

# 				#Optimization step 
# 				with tf.control_dependencies(self.extra_update_ops[i]): 
# 					#self.ao[i] = tf.train.MomentumOptimizer(self.learning_rate[i], momentum = 0.9, use_nesterov = True) 
# 					self.ao[i] = tf.train.AdamOptimizer(self.learning_rate[i]) 
# 					self.optimizer[i] = self.ao[i].minimize(self.cost[i])

# 				#Logger must be created after we have built the graph	
# 				params['filename']=cur_scope
# 				self.logger[i] = Logger(params, ceil(self.batch.train_length/self.batch_size)) 

# 		params['filename'] = self.base_name + "_Ensemble"
# 		self.ens_logger = Logger(params, ceil(self.batch.train_length/self.batch_size)) 
# 		params['filename'] = self.base_name
# 		#self.merged = tf.summary.merge_all()
# 		#self.writer = tf.summary.FileWriter('tb', graph = tf.get_default_session())

# 	def optimize(self, sess, feed_dict, epoch): 
# 		log_data = []
# 		ensemble_logits = None
# 		for i in range(self.ensemble_size): 
# 			log_data.append({})
# 			feed_dict[self.learning_rate[i]] = self.get_learning_rate(epoch)
# 			_, log_data[i]['train_cost'], log_data[i]['train_error'], logits, labels = sess.run([self.optimizer[i], self.cost[i], 
# 				self.error[i], self.logits[i], self.y], feed_dict = feed_dict)
# 			if ensemble_logits is None:
# 				ensemble_logits = logits
# 			else:
# 				ensemble_logits += logits
# 		ensemble_preds = np.argmax(ensemble_logits, axis = 1)
# 		ensemble_accuracy = np.sum(ensemble_preds == np.argmax(labels, axis = 1))/len(ensemble_preds)
# 		ce=0.
# 		for i in range(len(ensemble_logits)):
# 			#simplified form of cross entropy when there is the reference distribution is concentrated at a single term
# 			ce+= -np.sum(ensemble_logits[i]*labels[i]) + np.log(np.sum(np.exp(ensemble_logits[i])))
# 		log_data.append({})
# 		log_data[-1]['ensemble_train_error'] = 1.0 - ensemble_accuracy
# 		log_data[-1]['ensemble_train_cost'] = ce/len(ensemble_logits)
# 		return log_data


# 	def get_errors(self, sess, feed_dict, epoch): 
# 		feed_dict[self.training]  = False 
# 		log_data = []
# 		ensemble_logits = np.zeros((self.batch.valid_length, feed_dict[self.y].shape[1]))
# 		#We iterate over each of the models in the ensemble doing the entire validation set in the outer loop.
# 		#Switching the order would cause more cache mixes, but would simplify the code slightly.
# 		for i in range(self.ensemble_size): 
# 			feed_dict[self.learning_rate[i]] = self.get_learning_rate(epoch)

# 			cost = 0. 
# 			wrong_preds = 0. 

# 			#Go through the validation set in batches (to avoid memory overruns). 
# 			#Sum up the unaveraged error statistics 
# 			ix = 0
# 			for feed_dict[self.x], feed_dict[self.y] in self.batch.valid_batches(self.batch_size): 
# 				c, w, logits = sess.run([tf.reduce_sum(self.cost[i]), 
# 					tf.reduce_sum(tf.cast(self.incorrect_pred[i], tf.float32)), self.logits[i]], feed_dict = feed_dict) 
# 				cost += c*len(feed_dict[self.y])
# 				wrong_preds += w 
# 				ensemble_logits[ix:ix + len(logits)] += logits
# 				ix += len(logits)

# 			log_data.append({'test_cost' : cost / self.batch.valid_length, 'test_error' : wrong_preds / self.batch.valid_length})
# 			# log_data[i]['test_cost'], log_data[i]['test_error'], logits, labels = sess.run([self.cost[i], self.error[i],
# 				# self.logits[i], self.y], feed_dict = feed_dict)
# 			# if ensemble_logits is None:
# 			# 	ensemble_logits = logits
# 			# else:
# 			# 	ensemble_logits += logits
# 			# self.writer.add_summary(summary, epoch)
# 		ensemble_logits /= self.ensemble_size
# 		ensemble_preds = np.argmax(ensemble_logits, axis = 1)
# 		ensemble_accuracy = np.sum(ensemble_preds == np.argmax(self.batch.valid_labels, axis = 1))/len(ensemble_preds)

# 		ce=0.
# 		for i in range(self.batch.valid_length):
# 			#simplified form of cross entropy when there is the reference distribution is concentrated at a single term
# 			ce+= -np.sum(ensemble_logits[i]*self.batch.valid_labels[i]) + np.log(np.sum(np.exp(ensemble_logits[i])))

# 		log_data.append({})
# 		log_data[-1]['ensemble_test_error'] = 1.0 - ensemble_accuracy
# 		log_data[-1]['ensemble_test_cost'] = ce/self.batch.valid_length
# 		return log_data

# 	def log_batch(self, log_data):
# 		for i in range(self.ensemble_size):
# 			self.logger[i].batch_update(log_data[i]['train_error'], log_data[i]['train_cost'])
# 		self.ens_logger.batch_update(log_data[-1]['ensemble_train_error'], log_data[-1]['ensemble_train_cost'])

# 	def log_epoch(self, log_data, sess):
# 		for i in range(self.ensemble_size):
# 			self.logger[i].epoch_update(log_data[i]['test_error'], log_data[i]['test_cost'], None)
# 		self.ens_logger.epoch_update(log_data[-1]['ensemble_test_error'], log_data[-1]['ensemble_test_cost'], None)

# class CollabModel(EnsembleModel):
# 	def __init__(self, data, labels, params): 
# 		EnsembleModel.__init__(self, data, labels, params)
# 		#sum of costs functions = avg(sum of costs across models) effectively sum of avgs of costs
# 		self.collab_learning_rate = tf.placeholder(tf.float32, shape = []) 


# 		#get neighboring cross entropies
# 		self.ces = [None]*(self.ensemble_size-1)
# 		for i in range(self.ensemble_size-1):
# 			self.ces[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=
# 				self.scores[i+1], logits = self.logits[i]) + tf.nn.softmax_cross_entropy_with_logits(labels=
# 				self.scores[i], logits = self.logits[i+1]))

# 		self.total_cross_ent = tf.reduce_sum(tf.stack(self.ces))

# 		#get neighboring sum of square differences in logits
# 		self.sum_of_squares = [None]*(self.ensemble_size-1)
# 		for i in range(self.ensemble_size-1):
# 			self.sum_of_squares[i] = tf.reduce_mean((self.logits[i]-self.logits[i+1])**2)

# 		self.total_sum_of_squares = tf.reduce_sum(self.sum_of_squares)

# 		#get neighboring sum of absolute differences in logits
# 		self.sum_of_dists = [None]*(self.ensemble_size-1)
# 		for i in range(self.ensemble_size-1):
# 			self.sum_of_dists[i] = tf.reduce_mean(tf.abs(self.logits[i]-self.logits[i+1]))

# 		self.total_sum_of_dists = tf.reduce_sum(self.sum_of_dists)

# 		self.collab_weight = params['collab_weight']
# 		if params['collab_method'] == 'cross_ent':
# 			self.collab_cost = self.total_cross_ent*self.collab_weight
# 		elif params['collab_method'] ==  'L2':
# 			self.collab_cost = self.total_sum_of_squares*self.collab_weight
# 		elif params['collab_method'] == 'L1':
# 			self.collab_cost = self.total_sum_of_dists*self.collab_weight

# 		self.collab_ao = tf.train.AdamOptimizer(self.collab_learning_rate)
# 		self.collab_optimize = self.collab_ao.minimize(self.collab_cost)

# 	def optimize(self, sess, feed_dict, epoch): 
# 		log_data = []
# 		#Train each member of the collaboration to the training data
# 		for i in range(self.ensemble_size): 
# 			log_data.append({})
# 			feed_dict[self.learning_rate[i]] = self.get_learning_rate(epoch)
# 			_, log_data[i]['train_cost'], log_data[i]['train_error'] = sess.run([self.optimizer[i], self.cost[i], 
# 				self.error[i]], feed_dict = feed_dict)

# 		#Train the collaboration to come together
# 		feed_dict[self.collab_learning_rate] = self.get_learning_rate(epoch)
# 		_, batch_collab_cost = sess.run([self.collab_optimize, self.collab_cost], feed_dict = feed_dict)

# 		log_data.append({})
# 		log_data[-1]['collab_cost'] = batch_collab_cost
# 		return log_data

# 	def get_errors(self, sess, feed_dict, epoch): 
# 		feed_dict[self.training]  = False 
# 		log_data = []
# 		ensemble_logits = np.zeros((self.batch.valid_length, feed_dict[self.y].shape[1]))
# 		#We iterate over each of the models in the ensemble doing the entire validation set in the outer loop.
# 		#Switching the order would cause more cache mixes, but would simplify the code slightly.
# 		for i in range(self.ensemble_size): 
# 			feed_dict[self.learning_rate[i]] = self.get_learning_rate(epoch)

# 			cost = 0. 
# 			wrong_preds = 0. 

# 			#Go through the validation set in batches (to avoid memory overruns). 
# 			#Sum up the unaveraged error statistics 
# 			for feed_dict[self.x], feed_dict[self.y] in self.batch.valid_batches(self.batch_size): 
# 				c, w = sess.run([tf.reduce_sum(self.cost[i]), 
# 					tf.reduce_sum(tf.cast(self.incorrect_pred[i], tf.float32))], feed_dict = feed_dict) 
# 				cost += c*len(feed_dict[self.y])
# 				wrong_preds += w 

# 			log_data.append({'test_cost' : cost / self.batch.valid_length, 'test_error' : wrong_preds / self.batch.valid_length})

# 		feed_dict[self.collab_learning_rate] = self.get_learning_rate(epoch)
# 		log_data.append({})
# 		log_data[-1]['collab_cost'] = sess.run(self.collab_cost, feed_dict = feed_dict)
# 		return log_data

# 	def log_batch(self, log_data):
# 		for i in range(self.ensemble_size):
# 			self.logger[i].batch_update(log_data[i]['train_error'], log_data[i]['train_cost'])
# 		self.ens_logger.batch_update(0., log_data[-1]['collab_cost'])

# 	def log_epoch(self, log_data, sess):
# 		for i in range(self.ensemble_size):
# 			self.logger[i].epoch_update(log_data[i]['test_error'], log_data[i]['test_cost'], None)
# 		self.ens_logger.epoch_update(0., log_data[-1]['collab_cost'], sess)

def load_model(data, labels, filename, params, num_samples = 16):
    #reconstruct model without initializing parameters 
    model = Model.get_model(data, labels, params)

    saver = tf.train.Saver()
    with tf.Session() as sess:

        #restore parameters
        saver.restore(sess, filename)

        #Get samples data
        enc = OneHotEncoder(sparse = False) 
        OHElabels = enc.fit_transform(labels.reshape(-1,1)) 
        data = data.reshape([-1]+params['image_shape'])
        selections = np.random.randint(len(data), size = num_samples) 
        tdata, tlabels = data[selections], OHElabels[selections] 

        feed_dict = {}
        feed_dict[model.x] = tdata
        feed_dict[model.y] = tlabels
        feed_dict[model.training] = False
        preds = sess.run(model.preds, feed_dict = feed_dict)
        print("Predications:")
        print(preds)
        print("Labels:")
        print(labels)
        print("Bad Predictions:")
        print(preds[preds != labels[selections]])
        print("Correct Labels:")
        print(labels[selections][preds != labels[selections]])

#SAMPLE USAGE
#data, labels, image_shape = get_mnist()
# params = 'exact paramaters from model to reload' (saved in pickle)
#load_model(data[:-10000], labels[:-10000], 'log_data/shallow_and_thin-30-0.01380002498626709.ckpt', params, 
#    num_samples = 10000)

