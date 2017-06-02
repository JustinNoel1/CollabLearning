#import necessary modules
import os
import sys
import tarfile
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# import struct
# from array import array

#Our old friends
import tensorflow as tf
import numpy as np

#Progress bar
from tqdm import tqdm

#Our plotting and image packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from ggplot import *
# from IPython.display import display, Image
from scipy import ndimage

#Custom batch manager
from batch_manager import *


# ## Getting the data
def get_cifar(include_test = True, valid_size = 5000):
    """ Returns tuple containing the concatenated train and test_images, train and test_labels, the image size, 
    and the test length for the CIFAR10 dataset.
    """

    #pull data
    (train_images, train_labels), (test_images, test_labels) = tf.contrib.keras.datasets.cifar10.load_data()
    #concatenate and shuffle data
    if include_test:
        images = np.concatenate([train_images, test_images])
        labels = np.concatenate([train_labels, test_labels])
        test_length = len(test_labels)
    else:
        images = train_images
        labels = train_labels
        test_length = valid_size
    total_images, total_labels = randomize(images, labels)
    return images, labels, [32, 32, 3], test_length

def get_mnist():
    """ Returns tuple containing the concatenated train and test_images, train and test_labels, the image size, 
    and the test length for the MNIST dataset.
    """
    #pull data
    (train_images, train_labels), (test_images, test_labels) = tf.contrib.keras.datasets.mnist.load_data('datasets')
    #concatenate and shuffle
    total_images, total_labels = randomize(np.concatenate([train_images, test_images]), 
        np.concatenate([train_labels, test_labels]))

    image_shape = [28,28,1]
    return total_images, total_labels, image_shape, len(test_labels)

def get_notmnist(): 
    """ Returns tuple containing the concatenated train and test_images, train and test_labels, the image size,
    and the test length for the notMNIST dataset. This code has been taken from the tensorflow notMNIST example.
    """
    pickle_file = 'notMNIST_dataset/data.pkl'
    import pickle 
    try: 
        with open(pickle_file, 'rb') as f: 
            return pickle.load(f) 
    except Exception as e: 
        print('No pickle file yet.', pickle_file, ':', e)

    # from scipy.misc import imresize
    #Urls for datasets 
    notMNIST_url = 'http://yaroslavvb.com/upload/notMNIST/'

    #Target directories for downloads 
    notMNIST_dir = 'notMNIST_dataset'

    #Get notMNIST tar.gz files 
    #Extract notMNIST tar files and get folders 
    NM_test_filename = download_data('notMNIST_small.tar.gz', notMNIST_url, 8458043, notMNIST_dir) 
    test_folders = maybe_extract(NM_test_filename,notMNIST_dir) 
    NM_train_filename = download_data('notMNIST_large.tar.gz', notMNIST_url, 247336696, notMNIST_dir) 
    train_folders = maybe_extract(NM_train_filename, notMNIST_dir)
    train_datasets = maybe_pickle(train_folders, 50000) 
    test_datasets = maybe_pickle(test_folders, 1000) 

    train_size = 500000 
    test_size = 10000 
    _, _, train_dataset, train_labels = merge_datasets(train_datasets, train_size) 
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size) 

    print('Training:', train_dataset.shape, train_labels.shape) 
    print('Testing:', test_dataset.shape, test_labels.shape)

    train_images, train_labels = randomize(train_dataset, train_labels)
    test_images, test_labels = randomize(test_dataset, test_labels)
    total_images = np.concatenate([train_images,test_images])
    total_labels = np.concatenate([train_labels,test_labels])

    #Vectorize function for iterating over numpy array
    # def trans(x):
        # return imresize(x, (32,32))

    #Resize array
    # new_images = np.array([trans(x) for x in total_images])

    image_shape = [28,28,1] 
    with open(pickle_file, 'wb') as output:
            pickle.dump([total_images, total_labels, image_shape, test_size], output, pickle.HIGHEST_PROTOCOL)
    return total_images, total_labels, image_shape, test_size

def progress_hook(bar):
    """Prints progress bar as we download. Adapted from tqdm example."""
        
    #No static variables in ~ython. Sigh.
    last_block = [0]

    def inner(count=1, block_size=1, tsize=None):
        """
        count  : int, optional
            Number of blocks just transferred [default: 1].
        block_size  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            bar.total = tsize
        bar.update((count - last_block[0]) * block_size)
        last_block[0] = count
    
    return inner


def download_data(filename, url, expected_bytes, directory = 'datasets'):
    """Download filename from url into directory. Adapted from tensorflow example."""

    #Make directory if necessary
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    dest_filename = os.path.join(directory, filename)
    if not os.path.exists(dest_filename):
        print('Attempting to download:', filename) 
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as pbar:
            filename, _ = urlretrieve(url + filename, dest_filename, 
                                      reporthook=progress_hook(pbar))
            print('\nDownload Complete!')
    
    #Check the file
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
          'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


def maybe_extract(filename, force=False, data_root = 'notMNIST_dataset'):
  """Extracts the notMNIST data if necessary and returns a list of folders containing the data for each letter"""

  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(data_root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != 10:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        10, len(data_folders)))
  print(data_folders)
  return data_folders

def load_letter(folder, min_num_images): 
    """Load the data for a single letter label."""
    pixel_depth = 255.0  # Number of levels per pixel. 
    image_size = 28  # Pixel width and height. 
    image_files = os.listdir(folder) 
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32) 
    num_images = 0 
    for image in image_files: 
        image_file = os.path.join(folder, image)
        try: 
            image_data = ndimage.imread(image_file).astype(float)
            if image_data.shape != (image_size, image_size): 
                raise Exception('Unexpected image shape: %s' % str(image_data.shape)) 

            dataset[num_images, :, :] = image_data 
            num_images = num_images + 1 
        except IOError as e: 
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.') 

    dataset = dataset[0:num_images, :, :] 
    if num_images < min_num_images: 
        raise Exception('Many fewer images than expected: %d < %d' % 
            (num_images, min_num_images)) 
    return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False): 
    """Pickles the data for each letter if it has not already been pickled"""
    dataset_names = [] 
    for folder in data_folders: 
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename) 
        if os.path.exists(set_filename) and not force: 
            pass
        # You may override by setting force=True. 
        #print('%s already present - Skipping pickling.' % set_filename) 
        else: 
            print('Pickling %s.' % set_filename) 
            dataset = load_letter(folder, min_num_images_per_class) 
            try: 
                with open(set_filename, 'wb') as f: 
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL) 
            except Exception as e: 
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names

def make_arrays(nb_rows, img_size): 
    """Initializes numpy arrays"""
    if nb_rows: 
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32) 
        labels = np.ndarray(nb_rows, dtype=np.int32) 
    else: 
        dataset, labels = None, None 
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0): 
    """Merges the data from each of the letter classes

    Returns:
        tuple containing the training set, the training labels, the validation set, and the validation labels.
    """
    image_size = 28
    num_classes = len(pickle_files) 
    valid_dataset, valid_labels = make_arrays(valid_size, image_size) 
    train_dataset, train_labels = make_arrays(train_size, image_size) 
    vsize_per_class = valid_size // num_classes 
    tsize_per_class = train_size // num_classes 

    start_v, start_t = 0, 0 
    end_v, end_t = vsize_per_class, tsize_per_class 
    end_l = vsize_per_class+tsize_per_class 
    for label, pickle_file in enumerate(pickle_files): 
        try: 
            with open(pickle_file, 'rb') as f: 
                letter_set = pickle.load(f) 
                # let's shuffle the letters to have random validation and training set 
                np.random.shuffle(letter_set) 
                if valid_dataset is not None: 
                    valid_letter = letter_set[:vsize_per_class, :, :] 
                    valid_dataset[start_v:end_v, :, :] = valid_letter 
                    valid_labels[start_v:end_v] = label 
                    start_v += vsize_per_class 
                    end_v += vsize_per_class 

                train_letter = letter_set[vsize_per_class:end_l, :, :] 
                train_dataset[start_t:end_t, :, :] = train_letter 
                train_labels[start_t:end_t] = label 
                start_t += tsize_per_class 
                end_t += tsize_per_class 
        except Exception as e: 
            print('Unable to process data from', pickle_file, ':', e)
            raise
    
    return valid_dataset, valid_labels, train_dataset, train_labels
            
def randomize(dataset, labels): 
    """Shuffles the dataset and labels
    Returns:
        pair of shuffled dataset and shuffled labels
    """
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:] 
    shuffled_labels = labels[permutation] 
    return shuffled_dataset, shuffled_labels

def show_data(images, labels, label_names, title = "Sample Images", estimates = None, num_images = 5, shape = (28,28), 
              aug_params = None, output_dir = 'latex/images'):
    """Displays plots of images with their labels.

    Args:
        images (nparray) : array of images
        labels (nparray) : array of labels.
        label_names (dict) : a dictionary between the labels and the corresponding names to be printed
        title (str): title for figure
        estimages (list): list of predictions
        num_images (int): number of images to show 
        shape (int, int) or (int, int, int): shape of image to show
        distort (bool) : if true include display of distorted images.
    """ 
    
    #If we are distorting images, add another row to the figure and enlarge it
    if aug_params != None:
        idg = tf.contrib.keras.preprocessing.image.ImageDataGenerator(**aug_params)
        num_rows = 2 
        fig = plt.figure(figsize = (2*num_images, 4))
    else:
        num_rows = 1 
        fig = plt.figure(figsize = (2*num_images, 2.5))

    #for each image in our row
    for i in range(num_images):
        ix = np.random.randint(0, len(images))
        re = images[ix]
        ax = fig.add_subplot(num_rows, num_images, i+1) 
        ax.set_xlabel("Image = " + str(ix) + "\nLabel = " + label_names[int(labels[ix])])
        ax.imshow(re)
        ax.set_xticks([])
        ax.set_yticks([])
        #if we are distorting images, print another row of the same images, but distorted
        if aug_params != None:
            ax = fig.add_subplot(num_rows, num_images, num_images+i+1) 
            re = idg.random_transform(images[ix])
            ax.set_xlabel("Distorted Image = " + str(ix))
            ax.imshow(re)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(w_pad=4.0, h_pad=2.0)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fn = os.path.join(output_dir, title + '.png')
    plt.savefig(fn)

