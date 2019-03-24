import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
# file used to write preserve the results of the classfier
# confusion matrix and precision recall fscore matrix
#### This function will return the plot of the confusion matrix 
#1. input: confusion matrix and target names(class_name)
#2. output: plot of confusion matrix 
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    return plt



#### This function is generating the classification report
#1. input: ground_truth and predicted outputs
#2. output: dataframe containing the results

##saving the classification report
def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='macro'))
    avg.append(accuracy_score(y_true, y_pred, normalize=True))
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support','accuracy']
    list_all=list(metrics_summary)
    list_all.append(cm.diagonal())
    class_report_df = pd.DataFrame(
        list_all,
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-2] = total

    class_report_df['avg / total'] = avg

    return class_report_df.T



###this is the metric used for calculating the scores 
def calculate_score(y_true, y_pred, normalize=True, sample_weight=None):

    acc_list = []
    accuracy = []
    precision = []
    recall = []
    f1_score = []

    for i in range(y_true.shape[0]):

        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print(set_true)
        #print(set_pred)
        
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float( len(set_true.union(set_pred)) )
            temp_acc = len(set_true.intersection(set_pred))/float( len(set_true.union(set_pred)) )
            if len(set_pred) == 0:
                temp_pre = 0
            else:
                temp_pre = len(set_true.intersection(set_pred))/float( len(set_pred) )
            temp_rec = len(set_true.intersection(set_pred))/float( len(set_true))
            temp_f1 = 2*(len(set_true.intersection(set_pred)))/float(len(set_pred) + len(set_true) )
        #print('tmp_a*: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
        accuracy.append(temp_acc)
        precision.append(temp_pre)
        recall.append(temp_rec)
        f1_score.append(temp_f1)
        
        
    mean_hamming=hamming_loss(y_true, y_pred)
    mean_accuracy=np.mean(accuracy)
    mean_precision=np.mean(precision)
    mean_recall=np.mean(recall)
    mean_fscore=(2*mean_precision*mean_recall)/(mean_precision+mean_recall)
    return  mean_hamming,mean_accuracy,mean_precision,mean_recall,mean_fscore


def my_accuracy_score(y_train,y_train_pred):
    count=0
    for ele1,ele2 in zip(y_train,y_train_pred):
        if(list(ele1)==list(ele2)):
            count=count+1
    return count/y_train.shape[0]         