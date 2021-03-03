#---------------------------------------------
# params.py - AML Demo
# Nathan Huffman
# Combines params from file and command line
#---------------------------------------------

#---Parameters-----------------
epochs          = 1000
experiment      = 0
layers          = [8,8]
learning_rate   = 0.01

#---Imports-----------------
import argparse
import ast
    
# Handle arguments via cmd files
def handle_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--epochs', type=int, default=epochs,                       help='Number of epochs to train')
    parser.add_argument('--ex', type=int, default=experiment,                       help='Experiment number')
    parser.add_argument('--hidden', type=int, nargs='+', default=layers,            help='Hidden layer construction')
    parser.add_argument('--hidden2', type=ast.literal_eval, default=layers,         help='Alternative layer construction')
    parser.add_argument('-l', '--learn_rate', type=float, default=learning_rate,    help='Learning rate')
    parser.add_argument('-s', '--supercomputer', action='store_true',               help='Flag to run on supercomputer')
    parser.add_argument('-e', '--eval', action='store_true',                        help='Flag to run as evaluation')

    args = parser.parse_args()
    return args

print(handle_args())