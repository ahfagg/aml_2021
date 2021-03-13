#from hw4_base import *
import os
import fnmatch
import heapq
import sklearn
import itertools
import metrics
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

def read_all_rotations(dirname, filebase):
    '''
    Read results from dirname from files matching filebase
    :param dirname: Name of directory that contains the result pkl files
    :param filebase: String (with wildcards) that specifies the file(s) to be loaded
    :return: A list of experiment results files
    
    '''

    # The set of files in the directory
    files = fnmatch.filter(os.listdir(dirname), filebase)
    files.sort()
    results = []

    # Loop over matching files
    for f in files:
        fp = open("%s/%s"%(dirname,f), "rb")
        r = pickle.load(fp)
        fp.close()
        results.append(r)
    return results


def visualizeExperiment(dirname, filebase, metric='categorical_accuracy'):
    '''
    Generate the learning curves for a set of experiments for both the training and validation data sets.
    
    :param dirname: Name of directory that contains the result pkl files
    :param filebase: String (with wildcards) that specifies the file(s) to be loaded
    :param metric: Metric to be plotted.  The assumption is that the training set key is just metric and 
         the validation set key is "val_"+metric
    '''
    results = read_all_rotations(dirname, filebase)
    
    #Training Results
    for i, r in enumerate(results):
        plt.plot(r['history'][metric], label='Model {:d}'.format(i+1))
    plt.title('Training')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'lower right', prop={'size': 10})
    plt.show()
    
    #Validation Results
    for i, r in enumerate(results):
        plt.plot(r['history']['val_' + metric], label='Model {:d}'.format(i+1))
    plt.title('Validation')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'lower right', prop={'size': 10})
    plt.show()
    
    deep_network_accuracy = 0
    for r in results:
        deep_network_accuracy += np.average(heapq.nlargest(10, r['history']['val_'+metric]))
    print('Average Val Accuracy: ')
    print (deep_network_accuracy / len(results))
    
    print('Example Training Predictions: ')
    print(results[0]['predict_training'])
    
    print('Example Validation Predictions: ')
    print(results[0]['predict_validation'])


def visualizeConfusion(dirname, filebase, key_true='true_validation', key_predict='predict_validation'):
    '''
    Given a set of experiments:
    1. Plot the confusion matrix for each
    2. Compute and report the average AUC 
    
    :param dirname: Name of directory that contains the result pkl files
    :param filebase: String (with wildcards) that specifies the file(s) to be loaded
    :param key_true: The dictionary key that contains the true labels
    :param key_predict: The dictionary key that contains the model-predicted probabilities
    '''
    # Read all of the matching results
    result = read_all_rotations(dirname, filebase)
    
    auc = 0
    # Iterate over the results
    for r in result:
        # Compute AUC and the confusion matrix for each result
        preds_val = r[key_predict]
        outs_val = r[key_true]
        auc += metrics.multiclass_auc(outs_val, preds_val)
        metrics.generate_confusion_matrix(outs_val, preds_val, ['scissors', 'mugs', 'glasses'])
    print(auc / len(result))
    