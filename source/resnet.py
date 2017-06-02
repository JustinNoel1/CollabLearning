import tensorflow as tf

def resnet_layer(x, act, training, scope):
	"""
	This returns the output of a block of two 3x3 convolutions summed with a shortcut layer.

	Args:
	x : input tensor
	act : non-linearity
	training : whether or not we are training (used for batch normalization layers)
	"""
	with tf.name_scope(scope): 
		filters = x.shape[-1] 
		inits= tf.contrib.layers.variance_scaling_initializer() #He et al initializer 
		cinits = tf.constant_initializer() #zero initializer 

		# The order of operations is extremely important here. We need to pass the original input summed with our transformed input 
		# Center add learnable bias term. Scaling is unnecessary as we further compose with a linear function 
		o = tf.layers.batch_normalization(x, training=training, center = True, scale = False, trainable = True)#, name = "GlobalNormalize") 
		o = act(o) 
		# Any bias is wiped out by normalization 
		o = tf.layers.conv2d(o, filters, 3, kernel_initializer = inits, use_bias = False, padding = "SAME", activation = None) 
		o = tf.layers.batch_normalization(o, training=training, center = True, scale = False, trainable = True) 
		o = act(o) 
		o = tf.layers.conv2d(o, filters, 3, kernel_initializer = inits, bias_initializer = cinits, use_bias = True, activation = None, padding = "SAME") 
		return x+o

def resnet_smaller_layer(x, act, training, scope):
	"""
	This produces the output from a resnet layer where we have reduced the spatial dimensions by a factor of 2 and
	increased the feature dimension by 2.
	This will involve a shortcut calculated by a 2x2 valid padded convolution summed with a composition of two 3x3
	convolutions. This step requires the spatial dimensions to each be divisible by 2.

	Args:
	x : input tensor
	act : non-linearity
	training : whether or not we are training (used for batch normalization)
	scope : name of scope
	"""
	with tf.name_scope(scope): 
		filters = x.shape[-1] 
		inits= tf.contrib.layers.variance_scaling_initializer() 
		cinits = tf.constant_initializer() 

		o = tf.layers.batch_normalization(x, training=training, center = True, scale = False, trainable = True, name = scope + "BN") 
		o = act(o) 
		# Any bias is wiped out by the normalization 
		o = tf.layers.conv2d(x, filters*2, 3, strides = (2,2), use_bias = False, activation = None, kernel_initializer = inits, padding = "same") 
		o = tf.layers.batch_normalization(o, training=training, center = True, scale = False, trainable = True, name = scope +"BN2") 
		o = act(o) 
		o = tf.layers.conv2d(o, filters*2, 3, use_bias = False, kernel_initializer = inits, padding = "same") 

		#shortcut
		shortcut = tf.layers.conv2d(x, filters*2, 2, strides = (2,2), use_bias = True, kernel_initializer = inits, bias_initializer = cinits, padding = "VALID") 
		return o+shortcut

def resnet(o, act, training, times = 1, scope = 'Bricks', drop = True): 
	"""
	Returns the output of a resnet brick which consists of "times" number of blocks of two 3x3 convolutions. If
	drop is true then we will add a layer which reduces the spatial dimensions by a factor of 2. This latter step
	requires the spatial dimensions to be divisible by 2. 
	"""
	with tf.name_scope(scope):
		o = tf.contrib.layers.repeat(o, times, resnet_layer, act, training)
		if drop: 
			return resnet_smaller_layer(o,act,training, scope)
		else:
			return o

def deep_model(x, training, params = { 'act': tf.nn.relu, 'start_filter' : 16, 'num_bricks': 3, 'num_times' : 1}, scope = "Model"):
	"""
	Returns the output of a resnet network.

	Args:
	x : input tensor
	training : whether or not we are training (used for batch normalization).
	params : dictionary of parameters describing the model characteristics.
		This should contain 
		act : the non-linearity to use
		start_filter : the feature depth of the first convolutional layer
		num_bricks : the number of resnet bricks to include
		num_times : the size of the resnet bricks.
	scope : the name scope of the model
	"""
	with tf.name_scope(scope): 
		act = params['act'] 
		inits= tf.contrib.layers.variance_scaling_initializer() #He et al initializer 
		cinits = tf.constant_initializer() 

		#This just normalizes our input. 
		o = tf.layers.batch_normalization(x, training=training, center = False, scale = False, trainable = False)#, name="Normalize") 
		#Put input into the starting number of filters e.g. 32x32x3 32x32x16 
		o = tf.layers.conv2d(o, params['start_filter'], 3, padding = "same", use_bias = True, 
			kernel_initializer = inits, bias_initializer = cinits, activation = None) 

		#Repeat Resnet bricks 
		o = tf.contrib.layers.repeat(o, params['num_bricks']-1, resnet, act, training, times = params['num_times'], drop = True, scope=scope) 
		#No need for dimension reduction on last brick
		o = resnet(o, act, training, times = params['num_times'], drop = False)
		#global averaging layer
		o = tf.reduce_mean(o, axis = [1,2])
		#flatten to hidden layer and pass to logits 
		o = tf.contrib.layers.flatten(o) 
		return tf.layers.dense(o, 10, kernel_initializer = inits, bias_initializer = cinits)
