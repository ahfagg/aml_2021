import sklearn
import itertools
import matplotlib.pyplot as plt
import numpy as np

def get_auc(trues, preds):
    '''
    Compute the AUC of the ROC for binary comparison

    :param trues: True labels
    :param preds: Predicted probabilities
    :return: AUC
    '''
    
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(trues, preds)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc

def multiclass_auc(trues, preds):
    '''
    Calculates the average AUC of the ROC curve for each pair of classes in multiclass classification.

    :param trues: True labels
    :param preds: Predicted probabilities
    :return: Average AUC
    '''
    nClasses = trues.shape[1]
    
    # Identify the splits between each class in the data
    splits = np.sum(trues, axis=0, dtype='int')
    for i in range(1, nClasses):
        splits[i] += splits[i-1]
    splits = np.resize(splits, nClasses-1)
    
    # Separate the trues and predictions into each class
    trues_by_class = np.split(trues, splits)
    preds_by_class = np.split(preds, splits)
    
    # compute the AUC for each pair of classes
    auc = 0
    for (class1, class2) in itertools.combinations(range(nClasses), 2):
        binary_trues = np.concatenate([trues_by_class[class1], trues_by_class[class2]])[:, class1]
        binary_preds = np.concatenate([preds_by_class[class1], preds_by_class[class2]])[:, class1]     
        auc += get_auc(binary_trues, binary_preds)
        
    # return average AUC
    return auc / (nClasses * (nClasses-1) / 2)


def plot_confusion_matrix(cm, classes, normalize=False, 
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    PARAMS:
        cm: the confusion matrix
        classes: list of unique class labels
        normalize: boolean flag whether to normalize values
        title: figure title
        cmap: colormap scheme
    """
    # View percentages
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color='w' if cm[i, j] > thresh else 'k')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def generate_confusion_matrix(trues, preds, classes):
    '''
    This function generates and plots the confusion matrix for true value and predicted values
    
    classes: list of unique class labels
    '''
    nClasses = trues.shape[1]
    # format trues into single array containing class values
    # e.g. class value of [0, 1, 0] will be converted to 2
    new_trues = []
    for t in trues:
        for i in range(nClasses):
            if t[i] == 1:
                new_trues.append(i+1)
                break
    # threshold for prediction is highest probability class
    new_preds = np.argmax(preds, axis=1) + 1
    confusionMatrix = sklearn.metrics.confusion_matrix(new_trues, new_preds)
    plot_confusion_matrix(confusionMatrix, classes)
