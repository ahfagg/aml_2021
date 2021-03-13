'''
Author: Andrew H. Fagg
'''

import sys
tf_tools = "../../../../../tf_tools/"
sys.path.append(tf_tools + "metrics")
sys.path.append(tf_tools + "networks")
sys.path.append(tf_tools + "experiment_control")

from symbiotic_metrics import *
from cnn_classifier import *
from job_control import *
from core50 import *
import argparse
import pickle

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

#################################################################

def load_data_sets(directory_base, args, objects=None, condition_list=None, file_spec=None):
    '''
    Load an entire multi-class data set and create corresponding labels

    :param directory_base: Directory that contains the Core50 data set
    :param args: From argParse
    :param objects: A list of classes; each class contains a list of objects.  There can
       be any number of classes (2 and larger)
    :param condition_list: A list of Core50 conditions to load
    :param file_spec: The file pattern to match

    :return ins_train, outs_train, ins_val, outs_val
    '''
    if objects is None:
        objects = [['o11', 'o12', 'o13', 'o14', 'o15'], ['o41', 'o42', 'o43', 'o44', 'o45'], ['o26', 'o27', 'o28', 'o29', 'o30']]
        
    if condition_list is None:
        condition_list= ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']
        
    if file_spec is None:
        file_spec = '.*[0].png'
        
    ins_train = []
    outs_train = []
    ins_val = []
    outs_val = []
    
    # Iterate over classes
    for c in range(len(objects)):
        #
        print("#################", c)
        # Training data
        # Iterate over objects in a class
        train_list = []

        for i in range(args.Ntraining):
            train_list.append((i+args.rotation)%args.Nfolds)

        # List of specific objects
        obj = [objects[c][i] for i in train_list]
        print(obj)
        # Load these objects
        dat = load_multiple_image_sets_from_directories(directory_base, condition_list, obj, file_spec)
        # Append to our data set
        ins_train.append(dat)
        # Create the target vectors
        tmp = np.zeros((dat.shape[0], len(objects)))
        tmp[:, c] = 1; 
        outs_train.append(tmp)
        print('Train objects:', obj)
        
        # Validation data
        val_list = [(args.rotation+args.Nfolds-1)%args.Nfolds]
        # List of objects
        obj = [objects[c][i] for i in val_list]
        # Load these objects into memory
        dat = load_multiple_image_sets_from_directories(directory_base, condition_list, obj, file_spec)
        ins_val.append(dat)
        # Create the target vectors
        tmp = np.zeros((dat.shape[0], len(objects)))
        tmp[:, c] = 1; 
        outs_val.append(tmp)
        print('Validation objects:', obj)

    # Put the multi-fold training set together
    ins_train = np.concatenate(ins_train, axis=0)
    outs_train = np.concatenate(outs_train, axis=0)

    ins_val = np.concatenate(ins_val, axis=0)
    outs_val = np.concatenate(outs_val, axis=0)
    
    return ins_train, outs_train, ins_val, outs_val

def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='BMI Learner', fromfile_prefix_chars='@')
    parser.add_argument('-rotation', type=int, default=0, help='Cross-validation rotation')
    parser.add_argument('-epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('-dataset', type=str, default='/home/fagg/datasets/core50/core50_128x128', help='Data set directory')
    parser.add_argument('-Ntraining', type=int, default=4, help='Number of training folds')
    parser.add_argument('-exp_index', type=int, default=-1, help='Experiment index')
    parser.add_argument('-Nfolds', type=int, default=5, help='Maximum number of folds')
    parser.add_argument('-results_path', type=str, default='./results_hw4', help='Results directory')
    parser.add_argument('-hidden', nargs='+', type=int, default=[100, 5], help='Number of hidden units per layer (sequence of ints)')
    parser.add_argument('-conv_size', nargs='+', type=int, default=[3,5], help='Convolution filter size per layer (sequence of ints)')
    parser.add_argument('-conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('-pool', nargs='+', type=int, default=[2,2], help='Max pooling size (1=None)')
    parser.add_argument('-dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('-lrate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('-L2_regularizer', '-l2', type=float, default=None, help="L2 regularization parameter")
    parser.add_argument('-min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('-patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('-verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('-experiment_type', type=str, default=None, help="Experiment type")
    parser.add_argument('-nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('-batch', type=int, default=10, help="Training set batch size")
    
    return parser

#################################################################
def check_args(args):
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-1)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.dropout is None or (args.dropout > 0.0 and args.dropout < 1)), "Dropout must be between 0 and 1"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.L2_regularizer is None or (args.L2_regularizer > 0.0 and args.L2_regularizer < 1)), "L2_regularizer must be between 0 and 1"
    
def augment_args(args):
    '''
    Use the jobiterator to override the specified arguments based on the experiment index. 

    @return A string representing the selection of parameters to be used in the file name
    '''
    index = args.exp_index
    if(index == -1):
        return ""
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    if args.experiment_type is None:
        return ""
    elif args.experiment_type == "basic":
        p = {'rotation': range(5)}
    else:
        assert False, "Bad experiment type"
        
    
    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)
 
    
#################################################################

def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    '''
    # Hidden unit configuration
    hidden_str = '_'.join(str(x) for x in args.hidden)
    
    # Conv configuration
    conv_size_str = '_'.join(str(x) for x in args.conv_size)
    conv_filter_str = '_'.join(str(x) for x in args.conv_nfilters)
    pool_str = '_'.join(str(x) for x in args.pool)
    
    # Dropout
    if args.dropout is None:
        dropout_str = ''
    else:
        dropout_str = 'drop_%0.2f_'%(args.dropout)
        
    # L2 regularization
    if args.L2_regularizer is None:
        regularizer_str = ''
    else:
        regularizer_str = 'L2_%0.6f_'%(args.L2_regularizer)

        
    # Put it all together, including #of training folds and the experiment rotation
    return "%s/image_%s_hidden_%s_Csize_%s_Cfilters_%s_Pool_%s_%s%sntrain_%02d_rot_%02d"%(args.results_path, args.experiment_type,
                                                                                          hidden_str, 
                                                                                          conv_size_str,
                                                                                          conv_filter_str,
                                                                                          pool_str,
                                                                                          dropout_str,
                                                                                          regularizer_str,
                                                                                          args.Ntraining, args.rotation)


#################################################################
def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model
    
    @args Argparse arguments
    '''
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    # Modify the args in specific situations
    #augment_args(args)
    print(args.exp_index)
    args_str = augment_args(args)
    
    # Perform the experiment
    if(args.nogo):
        # No!
        return
    
    # Load data
    ins, outs, ins_validation, outs_validation = load_data_sets(args.dataset, args)
    image_size=(ins.shape[1], ins.shape[2])
    nchannels = ins.shape[3]

    print('TR:', ins.shape)
    print('V:', ins_validation.shape)
    
    # Network config
    # NOTE: this is very specific to our implementation of create_cnn_classifier_network()
    #   List comprehension and zip all in one place (ugly, but effective)
    dense_layers = [{'units': i} for i in args.hidden]
    conv_layers = [{'filters': f, 'kernel_size': (s,s), 'pool_size': (p,p), 'strides': (p,p)} if p > 1
                   else {'filters': f, 'kernel_size': (s,s), 'pool_size': None, 'strides': None}
                   for s, f, p, in zip(args.conv_size, args.conv_nfilters, args.pool)]
    
    print("Dense layers:", dense_layers)
    print("Conv layers:", conv_layers)
    
    # Build network
    model = create_cnn_classifier_network(image_size, nchannels,
                                         conv_layers=conv_layers,
                                          dense_layers=dense_layers,
                                          p_dropout=args.dropout,
                                          lambda_l2=args.L2_regularizer,
                                          lrate=args.lrate, n_classes=3)
    
    # Report if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())
    
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta)

    # Traning generator: produces a new mini-batch for each epoch
    generator = training_set_generator_images(ins, outs, batch_size=args.batch)

    # Learn
    history = model.fit(x=generator, epochs=args.epochs, steps_per_epoch=2,
                        use_multiprocessing=False, 
                        verbose=args.verbose>=2,
                        validation_data=(ins_validation, outs_validation), 
                        callbacks=[early_stopping_cb])
    
    # Generate log data
    results = {}
    results['args'] = args
    results['predict_training'] = model.predict(ins)
    results['predict_training_eval'] = model.evaluate(ins, outs)
    results['true_training'] = outs
    results['predict_validation'] = model.predict(ins_validation)
    results['predict_validation_eval'] = model.evaluate(ins_validation, outs_validation)
    results['true_validation'] = outs_validation
    #results['predict_testing'] = model.predict(ins_testing)
    #results['predict_testing_eval'] = model.evaluate(ins_testing, outs_testing)
    #results['folds'] = folds
    results['history'] = history.history
    
    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    fp = open("%s_results.pkl"%(fbase), "wb")
    pickle.dump(results, fp)
    fp.close()
    
    # Model
    model.save("%s_model"%(fbase))
    
    return model
#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    execute_exp(args)
