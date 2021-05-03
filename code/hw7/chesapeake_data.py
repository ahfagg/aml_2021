import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def load_file(file_base, fname):
    '''
    Load a single Chesapeake Watershed image and translate it into data ready for learning
    
    :param file_base: The directory that contains the file in question
    :param fname: Name of the file
    :return: ins = a set of network inputs (examples=1 x rows x cols x channels)
                    The channels include the visible light, NIR, and Landsat 8 channels
            mask = Building mask (not clear if it is useful)
            outs = Pixel labels (examples=1 x rows x cols).  Each label is an integer 0 ... 7
                            (index 15 in the data set is mapped to 0)
            weights = Mask for no data (index = 15)  (not clear that this is useful)
    '''
    fname = file_base + "/" + fname
    dat = np.load(fname)
    dat = dat['arr_0']
    dat = np.transpose(dat, [0, 2, 3, 1])
    
    # Labels: 8th element
    outs = dat[0, :, :, 8]
    #print(np.int_(outs))
    outs = np.int_(outs)
    
    # 15 = no data case
    weights = np.logical_not(np.equal(outs, 15)) * 1.0
    
    # Set all class 15 to class 0
    np.equal(outs, 15, where=0)
    
    # Image data
    images = dat[0, :, :, 0:8]/255.0
    
    # Landsat data
    # Unclear what the max is over the data set, but at least this gets us into the ballpark
    landsat = dat[0, :, :, 10:28]/4000.0
    
    # Building mask
    mask = dat[0, :, :, 28]
    
    ins = np.concatenate([images, landsat], axis=2)
    
    assert not np.isnan(ins).any(), "File contains NaNs (%s)"%(fname)
    assert np.min(outs) >= 0, "Labels out of bounds (%s, %d)"%(fname, np.min(outs))
    assert np.max(outs) < 7, "Labels out of bounds (%s, %d)"%(fname, np.max(outs))
    
    return ins, mask, outs, weights

    
def load_file_set(file_base, fnames):
    '''
    Load a list of Chesapeake Watershed images and translate into data ready for learning
    
    :param file_base: The directory that contains the file in question
    :param fname: List of file names to load
    :return: ins = a set of network inputs (examples x rows x cols x channels)
                    The channels include the visible light, NIR, and Landsat 8 channels
            mask = Building mask (not clear if it is useful)
            outs = Pixel labels (examples x rows x cols).  Each label is an integer 0 ... 7
                            (index 15 in the data set is mapped to 0)
            weights = Mask for no data (index = 15)  (not clear that this is useful)
    '''
    ins_all = []
    mask_all = []
    outs_all = []
    weights_all = []
    
    # Load each file
    for fname in fnames:
        ins, mask, outs, weights = load_file(file_base, fname)
        ins_all.append(ins)
        mask_all.append(mask)
        outs_all.append(outs)
        weights_all.append(weights)

    assert len(ins_all) > 0, "Nothing to load"
    
    # Compute shape for outputs
    sh = ins_all[0].shape
    sh_multi = (1, sh[0], sh[1], sh[2])
    sh_single = (1, sh[0], sh[1])
    
    # Concatenate the different structures together
    ins_all = [np.reshape(ins, newshape=sh_multi) for ins in ins_all]
    ins_all = np.concatenate(ins_all, axis = 0)
    
    mask_all = [np.reshape(mask, newshape=sh_single) for mask in mask_all]
    mask_all = np.concatenate(mask_all, axis = 0)
    
    outs_all = [np.reshape(outs, newshape=sh_single) for outs in outs_all]
    outs_all = np.concatenate(outs_all, axis = 0)
    
    weights_all = [np.reshape(weights, newshape=sh_single) for weights in weights_all]
    weights_all = np.concatenate(weights_all, axis = 0)
    
    return ins_all, mask_all, outs_all, weights_all

def load_files_from_dir(file_base, filt='-[1234]?'):
    '''
    Load all npz files from the specified directory that match the specified filter
    
    :param file_base: Directory that contains the npz file
    :param filt: Filter for the file groups.  
         File groups can be 0 ... 499 (with no leading zeros).  The default
         filter loads 500 examples from the training directory (of 5000).
         I suggest not trying to load all 5000 at once
         
    :return: ins = a set of network inputs (examples x rows x cols x channels)
                    The channels include the visible light, NIR, and Landsat 8 channels
            mask = Building mask (not clear if it is useful)
            outs = Pixel labels (examples x rows x cols).  Each label is an integer 0 ... 7
                            (index 15 in the data set is mapped to 0)
            weights = Mask for no data (index = 15)  (not clear that this is useful)
            
    '''
    files = [f for f in os.listdir(file_base) if re.match(r'.*%s.npz'%filt, f)]
    ins, mask, outs, weights = load_file_set(file_base, files)
    
    assert np.min(outs) >= 0, "Negative pixel class labels are not allowed"
    assert np.max(outs) < 7, "Illegal pixel class label (%d)"%(np.max(outs))
    
    
    print("Examples loaded:", ins.shape[0])
    
    return ins, mask, outs, weights 


def compute_class_weights(outs, div=100):
    '''
    Compute the weights for all of the classes.  The smallest class receives a weight
    of 1.  Other classes receive weights related to 1/c
    
    :param outs: a tensor of outputs
    :param div: a divisor that controls the sensitivity to very large imbalances.  
                The smaller this number is, the smaller the difference between 
                class weights
                
    NOTE: in the end, can't use this in model.fit(), sadly
    '''
    hist = np.bincount(outs.flatten())
    total = np.sum(hist)
    smallest = np.min(hist)+total/div
    
    weights = [smallest/(c+total/div) for c in hist]
    
    # Convert to a dict
    out={}
    for i,w in enumerate(weights):
        out[i] = w
    return hist, out

def pixel_class_to_image_class(outs):
    '''
    Convert pixel-level labels into class labels.  The classes are:
    0: water present
    1: Low vegetation present
    2: Barren land present
    3: Impervious-other present
    4: Impervious-road present 
    
    Note that the image-level classes are independent from one-another, so
    this becomes a binary classification problem (5 simultaneous binary
    classification problems)
    
    :param outs: Pixel-level classification (examples x rows x cols)
    :return: Image-level classification (examples x 5)
    
    Notes:
    - Water seems to be low frequency
    - Low vegetation is relatively high frequency
    - Barren land is very low frequency
    - The two types of impervious surfaces are highly correlated 
    '''
    sh = outs.shape
    ret = np.zeros((sh[0], 5))
    for i in range(outs.shape[0]):
        hist = np.bincount(outs[i,:,:].flatten())
        ret[i,0] = (hist[1] > 50)
        if(len(hist) >= 4 and hist[3] > 1000):
            ret[i,1] = 1
        if(len(hist) >= 5 and hist[4] > 50):
            ret[i,2] = 1
        if(len(hist) >= 6 and hist[5] > 500):
            ret[i,3] = 1
        if(len(hist) >= 7 and hist[6] > 500):
            ret[i,4] = 1
            
            
    labels=['water', 'low veg', 'barren', 'impervious', 'road']
    
    return ret, labels