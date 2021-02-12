import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
import re

import argparse
import pickle

# Tensorflow 2.0 way of doing things
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential

#################################################################
# Default plotting parameters
FONTSIZE = 18
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

#################################################################
def build_model(n_inputs, n_hidden, n_output, activation='elu', lrate=0.001):
    '''
    Construct a network with one hidden layer
    - Adam optimizer
    - MSE loss
    '''
    model = Sequential();
    model.add(InputLayer(input_shape=(n_inputs,)))
    model.add(Dense(n_hidden, use_bias=True, name="hidden", activation=activation))
    model.add(Dense(n_output, use_bias=True, name="output", activation=activation))
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    
    # Bind the optimizer and the loss function to the model
    model.compile(loss='mse', optimizer=opt)
    
    # Generate an ASCII representation of the architecture
    print(model.summary())
    return model

########################################################
def execute_exp(args):
    '''
    Execute a single instance of an experiment.  The details are specified in the args object
    
    '''

    ##############################
    # Run the experiment
    # Create training set: XOR
    ins = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outs = np.array([[0], [1], [1], [0]])
    
    model = build_model()

    # Callbacks
    #checkpoint_cb = keras.callbacks.ModelCheckpoint("xor_model.h5",
    #                                                save_best_only=True)

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=100,
                                                 restore_best_weights=True,
                                                 min_delta=.00001)

    # Training
    history = model.fit(x=ins, y=outs, epochs=args.epochs, verbose=False,
                        validation_data=(ins, outs),
                        callbacks=[early_stopping_cb])

    # Save the training history
    fp = 
    fp.close()

def display_learning_curve(fname):
    '''
    Display the learning curve that is stored in fname
    '''
    
    # Load the history file
    fp = open(fname, "rb")
    history = pickle.load(fp)
    fp.close()
    
    # Display
    plt.plot(history['loss'])
    plt.ylabel('MSE')
    plt.xlabel('epochs')

def display_learning_curve_set(base):
    '''
    Plot the learning curves for a set of results
    '''
    # Find the list of files in the local directory that match base_[\d]+.pkl
    files = [f for f in os.listdir('.') if re.match(r'%s_[0-9]+.pkl'%(base), f)]
    files.sort()
    
    # Iterate over the files
    for f in files:
        # Open and display each learning curve
        with open(f, "rb") as fp:
            history = pickle.load(fp)
            plt.plot(history['loss'])
            
    # Finish off the figure
    plt.ylabel('MSE')
    plt.xlabel('epochs')
    plt.legend(files)
    
def create_parser():
    '''
    Create a parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='XOR Learner')
    parser.add_argument('--exp', type=int, default=0, help='Experiment index')

    return parser

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Do the work
    execute_exp(args)
