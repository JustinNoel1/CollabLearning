import pickle
import os
import time
import tensorflow as tf 
import numpy as np
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
class Logger(object): 
    """The Logger class is responsible for logging information from training."""

    def __init__(self, params, batches_per_epoch = 1): 
        """Initializes Logger for logging information.

        Args:
            params : Model parameters
            batches_per_epoch : the number of batches each epoch will contain
        """
        filename = params['filename']
        self.filename = filename
        self.epochs = params['epochs']
        directory = params['data_dir']
        self.directory  = directory

        #Name of pickle file for saving data.
        self.picklename = os.path.join(directory, filename +".pkl")

        #Names of graph files to log
        self.error_graph_name = os.path.join(directory, filename +"_error.png")
        self.cost_graph_name = os.path.join(directory, filename +"_cost.png")

        self.params = params
        #Saver responsible for 
        self.saver = tf.train.Saver()

        #Initialize arrays of training data
        self.train_errors = np.zeros(self.epochs) 
        self.train_costs = np.zeros(self.epochs) 
        self.test_errors = np.zeros(self.epochs) 
        self.test_costs = np.zeros(self.epochs) 

        #Processing times
        self.start_time = time.time()
        self.end_time = time.time()

        #Running totals for batch training error
        self.batch_train_error = 0.0
        self.batch_train_cost = 0.0
        self.batches_per_epoch = batches_per_epoch

        #indexes current epoch
        self.ix = 0 

        #make log directory if necessary
        if not os.path.exists(directory): 
            os.makedirs(directory)

    def batch_update(self, train_error, train_cost):
        """Updates logs after running a batch"""

        #calculate average over batch
        self.batch_train_error = train_error
        self.batch_train_cost = train_cost

    def epoch_update(self, test_error, test_cost, sess): 
        """Update logs after an epoch"""
        #Print status  information 
        print("\nEpoch {} ({})".format(self.ix+1, self.filename))
        print("Train Loss = {:6f}".format(self.batch_train_cost))
        print("Train Error = {:6f}".format(self.batch_train_error*100))
        print("Test Loss = {:6f}".format(test_cost))
        print("Test Error = {:6f}".format(test_error*100))
        self.end_time=time.time()
        print("Time (Min) = {:6f}".format((self.end_time-self.start_time)/60.))

        #update data arrays
        self.train_costs[self.ix]=self.batch_train_cost
        self.train_errors[self.ix]=self.batch_train_error
        self.test_errors[self.ix]=test_error
        self.test_costs[self.ix]=test_cost

        #reset our increments
        self.batch_train_cost = 0.
        self.batch_train_error = 0.
        #increment our counter
        self.ix+=1 

        #Plot our graphs
        fig = plt.figure() 
        fig.suptitle("Errors")
        plt.plot(range(1,self.ix+1), 100.*self.train_errors[0:self.ix], 
            label = "Train Error Rate ({:%})".format(self.train_errors[self.ix-1]))
        plt.plot(range(1,self.ix+1), 100.*self.test_errors[0:self.ix], 
            label = "Test Error Rate ({:%})".format(self.test_errors[self.ix-1]))
        plt.xlabel("Epochs")
        plt.ylabel("Error %")
        plt.legend()
        plt.savefig(self.error_graph_name)
        plt.close()

        fig = plt.figure() 
        fig.suptitle("Loss")
        # plt.yscale('log')
        plt.plot(range(1,self.ix+1), self.train_costs[0:self.ix], 
            label = "Train Loss ({:f})".format(self.train_costs[self.ix-1]))
        plt.plot(range(1,self.ix+1), self.test_costs[0:self.ix], 
            label = "Test Loss ({:f})".format(self.test_costs[self.ix-1]))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.cost_graph_name)
        plt.close()

        #Save our selves!
        with open(self.picklename, 'wb') as output:
            pkl = {}
            pkl['train_errors'] = self.train_errors
            pkl['train_costs'] = self.train_costs
            pkl['test_errors'] = self.test_errors
            pkl['test_costs'] = self.test_costs
            pkl['start_time'] = self.start_time
            pkl['end_time'] = self.end_time
            pkl['params'] = self.params
            pickle.dump(pkl, output, pickle.HIGHEST_PROTOCOL)

        #Every 10 epochs store a checkpoint
        if (self.ix) % 10 == 0 and sess != None:
            self.saver.save(sess, os.path.join(self.directory, "{}-{}-{}.ckpt".format(self.filename, self.ix, test_error)))
