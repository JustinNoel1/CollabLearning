import numpy as np  
import pandas as pd 
import pickle
import os
import matplotlib.pyplot as plt

def build_dataframe(dataset):
    """
    Collects the pickles for the given dataset, unpickles them, and constructs a datagrame containing them.

    Args:
        dataset (str): Name of dataset.
    """
    import os
    #directory for pickles
    pickle_pantry = 'AWS8/collab_logs/'+dataset+'/collab/'

    #build column headers
    models = ['ST', 'M', 'Ens']
    methods = ['CE', 'L1', 'L2', 'Single']
    weights = [1.,0.1,0.01,0.0001]
    data = ['TrainCost', 'TrainError', 'TestCost', 'TestError']
    
    #Build multiindex
    arr = [ (m, me, w, d) for m in models 
           for me in methods 
           for w in weights
           for d in data]
    colnames = pd.MultiIndex.from_tuples(arr, names = ["Model", "Method", "Weight", "Data"])

    df = pd.DataFrame(columns = colnames)

    #Go through the pickles
    for file in os.listdir(pickle_pantry):
        #look at the pickles
        if file[-3:]=='pkl':
            with open(os.path.join(pickle_pantry,file), "rb") as handle:
                unpickle = pickle.load(handle)
                params = unpickle['params']
                #Determine which model the pickle is about, we can read this off from a specific character in the filename.
                if file[-5]=='e':
                    m = 'Ens'
                elif file[-5]=='0':
                    m = 'ST'
                else:
                    m = 'M'
                
                #Determine what type of collaboration method is being used
                #and set values accordingly
                if params['collab_method']=='None':
                    me = 'Single'
                    w = 1.
                elif params['collab_method']=='cross_ent':
                    me = 'CE'
                    w = round(params['collab_weight'],4)
                else:
                    me = params['collab_method']    
                    w = round(params['collab_weight'],4)

                #Save the data into the dataframe
                df[m,me,w,'TrainCost']=unpickle['train_costs']
                df[m,me,w,'TrainError']=100.*unpickle['train_errors']
                df[m,me,w,'TestCost']=unpickle['test_costs']
                df[m,me,w,'TestError']=100.*unpickle['test_errors']
           
    #Drop the missing entries coming from the single entries
    df.dropna(axis='columns', how='all', inplace=True)
    df.index = df.index+1
    return df.T

def graph_methods(ds, model, trunc, ylim, series, output_dir):
    """
    Produces graphs comparing each of the models for a fixed dataset and saves them.

    Args:
    ds (str) : The name of the dataset to graph.
    model (str) : Which model to graph (e.g. 'ST', 'M').
    trunc (slice) : Which epochs to graph.
    ylim (2-tuple of floats) : Set the limits of the y-axis so graphs have the same limits.
    series (str) : Which statistics to graph (e.g., 'TestError', 'TestCost', 'TrainError', 'TrainCost')
    output_dir (str) : Directory to save the graphs to.
    """
    df = build_dataframe(ds)
    #Get colormaps
    from matplotlib import cm
    cmd = {'CE' : cm.get_cmap('Blues_r'), 'L1': cm.get_cmap('Greens_r'), 'L2' : cm.get_cmap('Purples_r')}
    ylabel = {'TestError' : 'Test Error %', 
              'TestCost' : 'Test Cost',
              'TrainError' : 'Training Error %',
              'TrainCost' : 'Training Cost'}
    #Set figure size
    fig = plt.figure(figsize = (20, 6))

    for ix, me in enumerate(['CE', 'L2', 'L1']):
        ax = fig.add_subplot(1, 3, ix+1)
        ax.set_title(ds + ' ' + me)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel[series])
        ax.set_ylim(*ylim)
        for jx, y in enumerate([1.,0.1,0.01,0.0001]):
            ax.plot(df.loc[model,me,y,series].T[trunc], label=model+' ' + str(y), linewidth = 1+ jx, color = cmd[me](33*jx), alpha = 0.7)
        ax.plot(df.loc['ST','Single',1.,series].T[trunc], label='ST Single', linewidth = 2, color = cm.get_cmap('Greys_r')(100))
        ax.plot(df.loc['M','Single',1.,series].T[trunc], label='M Single', linewidth = 3, color =cm.get_cmap('Greys_r')(50))
        ax.plot(df.loc['Ens','Single',1.,series].T[trunc], label='Ens Single', linewidth = 4, color =cm.get_cmap('Greys_r')(0))
        ax.legend(loc = 1)
    fig.tight_layout(w_pad=4.0, h_pad=2.0) 

    #Make directory if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fn = os.path.join(output_dir, ds + '_' + model + '_' + series + '.png')
    plt.savefig(fn)

def graph_methods_for_all_datasets(trunc, series, output_dir):
    """
    Graphs the statistics from series for all of the datasets.

    Args:
        df (pandas dataframe) : The dataframe containing the data series.
        num_epochs (int) : How many epochs to graph (always includes last epoch)
        series (str) : Which statistics to graph (e.g., 'TestError', 'TestCost', 'TrainError', 'TrainCost')
        output_dir (str) : Directory to save the graphs to.
    """

    for model in ['ST', 'M']:
        for ds, ylim in [('MNIST', (0.4,1.)), ('notMNIST', (1.5,4.)), ('CIFAR10', (20.,30.))]:
            graph_methods(ds, model, trunc, ylim, series, output_dir)

def generate_tables(output_dir, series):
    """
    Generates tables of statistics in latex format and saves them to a file.

    Args:
        output_dir (str) : Directory to store file.
        series (str) : Which information to gather for the file. (e.g. 'TestCost')
    """
    from tabulate import tabulate
    
    label = {'TestError' : 'Test Error %', 'TestCost' : 'Test Cost', 'TrainError' : 'Train Error %', 'TrainCost' : 'Train Cost'}

    #Make the directory if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #For each datset
    for ds in ['MNIST', 'notMNIST', 'CIFAR10']:
        #grab our dataframe
        df = build_dataframe(ds)
        fn = os.path.join(output_dir, ds+'_' + series + '_table.tex')
        with open(fn, 'w') as file:
            for mod in ['ST', 'M', 'Ens']:
                #grab the relevant subframe
                relevant_df = df.T[-1:].T.loc[mod, :, :, series]
                #Write the table
                file.write(tabulate(relevant_df, headers = [mod + ' Model', label[series]], tablefmt = 'latex'))

def analyze_model(ds, method, weight, output_dir):
    """
    Produces graphs comparing each of the models for a fixed dataset and saves them.

    Args:
        ds (str) : The name of the dataset to graph.
        model (str) : Which model to graph (e.g. 'ST', 'M').
        weight (float) : Which collaboration weight to consider
        output_dir (str) : Directory to save the graphs to.
    """
    df = build_dataframe(ds)
    #Get colormaps
    from matplotlib import cm

    #Set figure size
    fig = plt.figure(figsize = (15, 6))

    #Add the error plots
    ax = fig.add_subplot(1,2,1)
    ax.set_title(ds + ' ' + method + ' ' + str(weight) + ' Error rate')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error rate %')
    al = 0.3
    ylims = {'MNIST' : (0., 2.), 'notMNIST' : (1.1, 11.), 'CIFAR10' : (0., 60.) }
    ax.set_ylim(ylims[ds])
    ax.plot(df.loc['ST',method,weight,'TrainError'].T, '--',  label='ST Collab Train', linewidth = 2, color = cm.get_cmap('Purples_r')(100))
    ax.plot(df.loc['ST',method,weight,'TestError'].T, label='ST Collab Test', linewidth = 2, color = cm.get_cmap('Purples_r')(100))
    ax.plot(df.loc['M',method, weight,'TrainError'].T, '--', label='M Collab Train', linewidth = 2, color = cm.get_cmap('Purples_r')(50))
    ax.plot(df.loc['M',method, weight,'TestError'].T, label='M Collab Test', linewidth = 3, color = cm.get_cmap('Purples_r')(50))
    ax.plot(df.loc['Ens',method, weight,'TrainError'].T, '--', label='Ens Collab Train', linewidth = 4, color = cm.get_cmap('Purples_r')(25))
    ax.plot(df.loc['Ens',method, weight,'TestError'].T, label='Ens Collab Test', linewidth = 4, color = cm.get_cmap('Purples_r')(25))      
    ax.plot(df.loc['ST','Single',1.,'TrainError'].T, '--', alpha = al,  label='ST Single Train', linewidth = 2, color = cm.get_cmap('Greys_r')(100))
    ax.plot(df.loc['ST','Single',1.,'TestError'].T, alpha = al, label='ST Single Test', linewidth = 2, color = cm.get_cmap('Greys_r')(100))
    ax.plot(df.loc['M','Single',1.,'TrainError'].T, '--', alpha = al, label='M Single Train', linewidth = 3, color = cm.get_cmap('Greys_r')(50))
    ax.plot(df.loc['M','Single',1.,'TestError'].T, alpha = al, label='M Single Test', linewidth = 3, color = cm.get_cmap('Greys_r')(50))
    ax.plot(df.loc['Ens','Single',1.,'TrainError'].T, '--', alpha = al, label='Ens Single Train', linewidth = 4, color = cm.get_cmap('Greys_r')(0))
    ax.plot(df.loc['Ens','Single',1.,'TestError'].T, alpha = al, label='Ens Single Test', linewidth = 4, color = cm.get_cmap('Greys_r')(0))   
    ax.legend(loc=1)

    #Add the cost plots
    ax = fig.add_subplot(1,2,2)
    ax.set_title(ds + ' ' + method + ' ' + str(weight) + ' Cost')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost')
    ylims = {'MNIST' : (0., 0.15), 'notMNIST' : (0.05, 0.4), 'CIFAR10' : (0., 2.5) }
    ax.set_ylim(ylims[ds])
    ax.plot(df.loc['ST',method,weight,'TrainCost'].T, '--',  label='ST Collab Train', linewidth = 2, color = cm.get_cmap('Purples_r')(100))
    ax.plot(df.loc['ST',method,weight,'TestCost'].T, label='ST Collab Test', linewidth = 2, color = cm.get_cmap('Purples_r')(100))
    ax.plot(df.loc['M',method, weight,'TrainCost'].T, '--', label='M Collab Train', linewidth = 3, color = cm.get_cmap('Purples_r')(50))
    ax.plot(df.loc['M',method, weight,'TestCost'].T,  label='M Collab Test', linewidth = 3, color = cm.get_cmap('Purples_r')(50))
    ax.plot(df.loc['Ens',method, weight,'TrainCost'].T, '--', label='Ens Collab Train', linewidth = 4, color = cm.get_cmap('Purples_r')(25))
    ax.plot(df.loc['Ens',method, weight,'TestCost'].T,  label='Ens Collab Test', linewidth = 4, color = cm.get_cmap('Purples_r')(25))      
    ax.plot(df.loc['ST','Single',1.,'TrainCost'].T, '--', alpha = al,  label='ST Single Train', linewidth = 2, color = cm.get_cmap('Greys_r')(100))
    ax.plot(df.loc['ST','Single',1.,'TestCost'].T, alpha = al, label='ST Single Test', linewidth = 2, color = cm.get_cmap('Greys_r')(100))
    ax.plot(df.loc['M','Single',1.,'TrainCost'].T, '--', alpha = al, label='M Single Train', linewidth = 3, color = cm.get_cmap('Greys_r')(50))
    ax.plot(df.loc['M','Single',1.,'TestCost'].T, alpha = al, label='M Single Test', linewidth = 3, color = cm.get_cmap('Greys_r')(50))
    ax.plot(df.loc['Ens','Single',1.,'TrainCost'].T, '--', alpha = al, label='Ens Single Train', linewidth = 4, color = cm.get_cmap('Greys_r')(0))
    ax.plot(df.loc['Ens','Single',1.,'TestCost'].T, alpha = al, label='Ens Single Test', linewidth = 4, color = cm.get_cmap('Greys_r')(0))   
    ax.legend(loc=1)
    fig.tight_layout(w_pad=4.0, h_pad=2.0) 

    #Make directory if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fn = os.path.join(output_dir, ds + '_FINAL_' + method + '_' + str(weight) + '_Error.png')
    plt.savefig(fn) 
for ds in ['MNIST', 'notMNIST', 'CIFAR10']:
    analyze_model(ds, 'L2', 0.01, os.path.join('latex', 'graphs'))
graph_methods_for_all_datasets(slice(-10,None,1),'TestError', os.path.join('latex', 'graphs'))          
graph_methods_for_all_datasets(slice(-10,None,1),'TestCost', os.path.join('latex', 'graphs'))          

#generate_tables(os.path.join('latex', 'tables'), 'TestError')
#generate_tables(os.path.join('latex', 'tables'), 'TestCost')