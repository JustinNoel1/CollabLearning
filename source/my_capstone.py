# ## Importing necessary modules

#Our old friends
import numpy as np
import tensorflow as tf

#Our Model class
from Model import *
#Collect datasets
from datasets import *

#Set random seed
np.random.seed(912)

def train_model(data, labels, params):
    """Train a model.

    Args:
        data (numpy array): all of the data including the test set
        labels (numpy array): all of the corresponding labels
        params (dict): a dictionary of model parameters
    """
    #reset_graph
    tf.set_random_seed(912)
    tf.reset_default_graph()
    #Construct the model graph
    model = Model.get_model(data, labels, params)
   
    with tf.Session() as sess: 
        #Set the tensorflow random seed
        tf.set_random_seed(912)
        model.train(sess)

def get_num_model_params(data, labels, params):
    """
    Prints the number of trainable parameters in a model.
    Args:
        data (np array) : dataset to construct a model with
        labels (np array) : labels to construct a model with
        params (dict) : dictionary for the parameters defining the model.
    """

    #reset_graph
    tf.reset_default_graph()

    #Build the model
    model = Model.get_model(data, labels, params)

    with tf.Session() as sess: 
        #initialize variables
        sess.run(tf.global_variables_initializer()) 
        total_parameters = 0 
        print("Filename {}".format(params['filename']))
        for variable in tf.trainable_variables(): 
            # shape is an array of tf.Dimension 
            shape = variable.get_shape() 
            variable_parametes = 1 
            for dim in shape: 
                variable_parametes *= dim.value 
                total_parameters += variable_parametes 
        print("Total params: {}".format(total_parameters))

def train_single_models(total_images, total_labels, image_shape, test_length, aug_params, data_dir, base_name):
    """Trains a collection of individual models according to hardcoded parameters.

    Args:
        total_images (numpy array): all images including the test set
        total_labels (numpy array):  all labels for the images
        image_shape (length 3 list): the shape of the images
        test_length (int): number of samples to be set aside for testing
        aug_params (dict): dictionary of data augmentation settings
        data_dir (str): directory to store data
        base_name (str): filename prefix to indicate the dataset
    """
    #Testing parameters 
    params = { 'act': tf.nn.relu, 
        'data_dir' : os.path.join(data_dir, base_name, 'single'),
        'start_filter' : 2, 
        'num_bricks': 3, 
        'num_times' : 2, 
        'batch_size' : 64, 
        'epochs' : 11, 
        'image_shape' : image_shape, 
        'learning_rate' : 0.5**10, 
        'data_augmentation' : None, 
        'truncate_data' : True, 
        'valid_num' : test_length, 
        'shuffle' : True, 
        'filename' : base_name + '_single_' + 'test', 
        'regularization_rate' : 0.1**4,
        'model_type' : 'single'} 
    #Run test single model 
    #train_model(total_images, total_labels, params) 

    if base_name != 'notMNIST':
        params['epochs'] = 30 
    else:
        params['epochs'] = 15 
    params['truncate_data'] = False 
    params['data_augmentation'] = aug_params

    #ST model parameters
    params['start_filter'] = 8 
    params['num_bricks'] = 3 
    params['num_times'] = 1 
    params['filename'] =  base_name + '_single_' + 'shallow_and_thin'
    #Run shallow and thin single model 
    #train_model(total_images, total_labels, params) 

    #New medium model params 
    params['start_filter'] = 16 
    params['num_bricks'] = 3 
    params['num_times'] = 2 
    params['filename'] =  base_name + '_single_' + 'medium_'
    #Run medium model
    #train_model(total_images, total_labels, params) 

    #New deep model params
    params['num_times'] = 3 
    params['filename'] =  base_name + '_single_data_aug' + '3_times_'
    train_model(total_images, total_labels, params)

def train_ensemble_models(total_images, total_labels, image_shape, test_length, aug_params, data_dir, base_name):
    """Trains a collection of ensemble models according to hardcoded parameters

    Args:
        total_images (numpy array): all images including the test set
        total_labels (numpy array):  all labels for the images
        image_shape (length 3 list): the shape of the images
        test_length (int): number of samples to be set aside for testing
        aug_params (dict): dictionary of data augmentation settings
        data_dir (str): directory to store data
        base_name (str): filename prefix to indicate the dataset
    """

    #Testing parameters 
    test_params = { 'act': tf.nn.relu, 
        'data_dir' : os.path.join(data_dir, base_name, 'ensemble'),
        'start_filter' : 4, 
        'num_bricks': 3, 
        'num_times' : 2, 
        'batch_size' : 64, 
        'epochs' : 11, 
        'image_shape' : image_shape, 
        'learning_rate' : 0.5**10, 
        'data_augmentation' : None, 
        'truncate_data' : True, 
        'valid_num' : test_length, 
        'shuffle' : False, 
        'filename' : base_name + '_ensemble_' + 'test', 
        'regularization_rate' : 0.1**4,
        'model_type' : 'ensemble'}

    #2x test params
    #No data augmentation on test. The point is to check convergence and with a truncated dataset augmentation will
    #hurt convergence
    params = test_params.copy() #has the same general settings even if we don't use most of the parameters because 
                                         #they are passed in the list of params
    params['list_of_params'] = [test_params]*2
    #rain_model(total_images, total_labels, params) 

    #Train notMNIST for less time
    if base_name != 'notMNIST':
        params['epochs'] = 30 
    else:
        params['epochs'] = 15 

    params['truncate_data'] = True 
    params['data_augmentation'] = None 
    st_params = params.copy()
    st_params['start_filter'] = 8 
    st_params['num_bricks'] = 3 
    st_params['num_times'] = 1 

    medium_params = st_params.copy()
    #New medium model params 
    medium_params['start_filter'] = 16 
    medium_params['num_bricks'] = 3 
    medium_params['num_times'] = 2 

    #Add prefix to prevent overwriting log files 
    params['filename'] = base_name + '_mixed_ensemble_' + 'shallow_and_thin_and_medium'

    #Mixed ensemble
    params['list_of_params'] = [st_params, medium_params]
    #Train ensemble
    assert(params['epochs']==30)
    train_model(total_images, total_labels, params) 

def train_collab_models(total_images, total_labels, image_shape, test_length, aug_params, data_dir, base_name):
    """Trains a collection of collaborative and traditional ensembles according to hardcoded parameters

    Args:
        total_images (numpy array): all images including the test set
        total_labels (numpy array):  all labels for the images
        image_shape (length 3 list): the shape of the images
        test_length (int): number of samples to be set aside for testing
        aug_params (dict): dictionary of data augmentation settings
        data_dir (str): directory to store data
        base_name (str): filename prefix to indicate the dataset
    """   

    #Testing parameters 
    test_params = { 'act': tf.nn.relu, 
        'data_dir' : os.path.join(data_dir, base_name, 'collab'),
        'start_filter' : 4, 
        'num_bricks': 3, 
        'num_times' : 2, 
        'batch_size' : 64, 
        'epochs' : 11, 
        'image_shape' : image_shape, 
        'learning_rate' : 0.5**10, 
        'data_augmentation' : None, 
        'truncate_data' : True, 
        'valid_num' : test_length, 
        'shuffle' : False, 
        'filename' : base_name + '_collab_' + 'test', 
        'regularization_rate' : 0.1**4,
        'model_type' : 'collab'}

    #No data augmentation on test. The point is to check convergence and with a truncated dataset augmentation will
    #hurt convergence
    params = test_params.copy() #has the same general settings even if we don't use most of the parameters because 
                                         #they are passed in the list of params
    # params['list_of_params'] = [test_params]*3
    # train_model(total_images, total_labels, params) 
    
    params['data_augmentation'] = None 
    st_params = test_params.copy()
    st_params['start_filter'] = 8 
    st_params['num_bricks'] = 3 
    st_params['num_times'] = 1 

    medium_params = st_params.copy()
    #New medium model params 
    medium_params['start_filter'] = 16 
    medium_params['num_bricks'] = 3 
    medium_params['num_times'] = 2 

    #Add prefix to prevent overwriting log files 
    params['collab_method'] = 'None'
    params['collab_weight'] = 0.1**0
    params['filename'] = base_name + '_collab_' + 'truncated_test_ensemble_0.1x0'
    #Mixed collab
    params['list_of_params'] = [st_params, medium_params]
    if base_name != 'notMNIST':
        params['epochs'] = 30 
    else:
        params['epochs'] = 15 

    #Test run with truncated data
    #train_model(total_images, total_labels, params)   
    params['collab_method'] = 'L2'
    params['collab_weight'] = 0.1**2 
    params['filename'] = base_name + '_collab_' + 'truncated_test_L2_0.1x2'
    train_model(total_images, total_labels, params)   

    # params['collab_method'] = 'L2'
    # params['collab_weight'] = 0.1**2
    # params['filename'] = base_name + '_collab_' + 'truncated_test_L2_0.1x0'
    # train_model(total_images, total_labels, params)   

    # params['collab_method'] = 'L1'
    # params['collab_weight'] = 0.1**2
    # params['filename'] = base_name + '_collab_' + 'truncated_test_L1_0.1x0'
    # train_model(total_images, total_labels, params)   

    #First real run
    params['truncate_data'] = False 
    #Run ensemble
    params['collab_method'] = 'None'
    params['filename'] = base_name + '_collab_' + 'shallow_and_thin_and_medium_ensemble'
    train_model(total_images, total_labels, params)   

    #Run collab models
    params['collab_weight'] = 0.1**0
    params['collab_method'] = 'cross_ent'
    params['filename'] = base_name + '_collab_' + 'shallow_and_thin_and_medium_CE_0.1x0'
    train_model(total_images, total_labels, params)   

    params['filename'] = base_name + '_collab_' + 'shallow_and_thin_and_medium_L2_0.1x0'
    params['collab_method'] = 'L2'
    train_model(total_images, total_labels, params)   

    params['filename'] = base_name + '_collab_' + 'shallow_and_thin_and_medium_L1_0.1x0'
    params['collab_method'] = 'L1'
    train_model(total_images, total_labels, params)   

    #Run all of the models again with another collaboration weight
    params['collab_weight'] = 0.1**1
    params['collab_method'] = 'cross_ent'
    params['filename'] = base_name + '_collab_' + 'shallow_and_thin_and_medium_CE_0.1x1'
    train_model(total_images, total_labels, params)   

    params['filename'] = base_name + '_collab_' + 'shallow_and_thin_and_medium_L2_0.1x1'
    params['collab_method'] = 'L2'
    train_model(total_images, total_labels, params)   

    params['filename'] = base_name + '_collab_' + 'shallow_and_thin_and_medium_L1_0.1x1'
    params['collab_method'] = 'L1'
    train_model(total_images, total_labels, params)      

def train_models(total_images, total_labels, image_shape, test_length, aug_params, data_dir, base_name):
    """This trains the single, ensemble, and collaborative models according to the passed values."""
    train_collab_models(total_images, total_labels, image_shape, test_length, aug_params, data_dir, base_name)
    #train_ensemble_models(total_images, total_labels, image_shape, test_length, aug_params, data_dir, base_name)
    #train_single_models(total_images, total_labels, image_shape, test_length, aug_params, data_dir, base_name)

def train_mnist(data_dir): 
    """Train on the MNIST dataset and generate sample image file.

    Args:
        data_dir(str) : directory to store the log data.
    """

    #set data augmentation params
    aug_params =   { 
        'rotation_range' : 00.0, #\pm rotation in degrees 
        'width_shift_range' : 0.125,#\pmFraction of image to shift 
        'height_shift_range' : 0.125,#\pmFraction of image to shift 
        'shear_range' : 0.0, #\pm Shear angle in radians 
        'zoom_range' : 0.0, 
        'channel_shift_range' : 0.0,#\pm uniform shift in values in each channel 
        'fill_mode' : 'nearest', 
        'horizontal_flip' : False, 
        'vertical_flip' : False, 
        'rescale' : None}

    total_images, total_labels, image_shape, test_length = get_mnist() 
    label_names = { 0 : '0', 1 : '1', 2 : '2', 3 : '3', 4 : '4', 5 : '5', 6 : '6', 7 : '7', 8 : '8', 9 : '9'}
    show_data(total_images, total_labels, label_names, shape = image_shape, title = "MNIST_Sample_Images", 
                  aug_params = None, output_dir = os.path.join('..', 'latex', 'images'))
    train_models(total_images, total_labels, image_shape, test_length, aug_params, data_dir, 'MNIST')

def train_notmnist(data_dir): 
    """Train on the notMNIST dataset and generate sample image file.

    Args:
        data_dir(str) : directory to store the log data.
    """   

    #set data augmentation params
    aug_params =   { 
        'rotation_range' : 00.0, #\pm rotation in degrees 
        'width_shift_range' : 0.125,#\pmFraction of image to shift 
        'height_shift_range' : 0.125,#\pmFraction of image to shift 
        'shear_range' : 0.0, #\pm Shear angle in radians 
        'zoom_range' : 0.0, 
        'channel_shift_range' : 0.0,#\pm uniform shift in values in each channel 
        'fill_mode' : 'nearest', 
        'horizontal_flip' : False, 
        'vertical_flip' : False, 
        'rescale' : None}

    total_images, total_labels, image_shape, test_length = get_notmnist() 
    label_names = { 0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 5 : 'F', 6 : 'G', 7 : 'H', 8 : 'I', 9 : 'J'}
    show_data(total_images, total_labels, label_names, shape = image_shape, title = "notMNIST_Sample_Images", 
              aug_params = None, output_dir = os.path.join('..', 'latex', 'images'))
    train_models(total_images, total_labels, image_shape, test_length, aug_params, data_dir, 'notMNIST')

def train_cifar10(data_dir): 
    """Train on the cifar10 dataset

    Args:
        data_dir(str) : directory to store the log data.
    """   

    #set data augmentation params
    aug_params =   { 
        'rotation_range' : 00.0, #\pm rotation in degrees 
        'width_shift_range' : 0.125,#\pmFraction of image to shift 
        'height_shift_range' : 0.125,#\pmFraction of image to shift 
        'shear_range' : 0.0, #\pm Shear angle in radians 
        'zoom_range' : 0.0, 
        'channel_shift_range' : 0.0,#\pm uniform shift in values in each channel 
        'fill_mode' : 'nearest', 
        'horizontal_flip' : True, 
        'vertical_flip' : False, 
        'rescale' : None}

    label_names = { 0 : 'Airplane', 1 : 'Automobile', 2 : 'Bird', 3 : 'Cat', 4 : 'Deer', 5 : 'Dog', 6 : 'Frog', 7 : 'Horse', 8 : 'Ship', 9 : 'Truck'}
    total_images, total_labels, image_shape, test_length = get_cifar() 
    show_data(total_images, total_labels, label_names, shape = image_shape, title = "CIFAR10_Sample_Images",
                      aug_params = None, output_dir = os.path.join('..', 'latex', 'images'))
    train_models(total_images, total_labels, image_shape, test_length, aug_params, data_dir, 'CIFAR10')

#Train our models
train_mnist('collab_logs')
train_cifar10('collab_logs')
train_notmnist('collab_logs')
