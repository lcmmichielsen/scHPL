# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:48:26 2019

@author: Lieke
"""

import numpy as np
from newick import *
import pandas as pd

def hierarchical_F1(true_labels, pred_labels, tree):
    '''
    Calculate the hierarchical F1-score
    
    Parameters
    ----------
    true_labels: vector with the true labels 
    pred_labels: vector with the predicted labels
    tree: classification tree used to predict the labels
        
    Return
    ------
    hF1: hierarchical F1-score
    '''
    
    sum_p = 0
    sum_t = 0
    sum_o = 0
        
        
    for i in range(len(true_labels)):
        true_lab = true_labels[i]
        pred_lab = pred_labels[i]
        
        found = 0
        
        set_true = []
        set_pred = []
        
        for n in tree[0].walk('postorder'):
            if(n.name == true_lab):
                found += 1
                set_true.append(n.name)
                a = n.ancestor
                while(a != None):
                    set_true.append(a.name)
                    a = a.ancestor
                
                if found == 2:
                    break
                    
            if(n.name == pred_lab):
                found += 1
                set_pred.append(n.name)
                a = n.ancestor
                while(a != None):
                    if(a.name == true_lab):
                        set_pred = []
                    set_pred.append(a.name)
                    a = a.ancestor

                if found == 2:
                    break
        
        common = len(np.intersect1d(set_pred, set_true))
        pred_len = len(set_pred)
        true_len = len(set_true)
        
        
        sum_p += pred_len - 1 # -1 to remove root
        sum_t += true_len - 1 # -1 to remove root
        sum_o += common - 1 # -1 to remove root

    hP = sum_o/sum_p
    hR = sum_o/sum_t      
                
    hF1 = (2 * hP * hR)/(hP + hR)
    
    
    return hF1


def confusion(y_true, y_pred):
    '''
    Construct a confusion matrix
    '''
    
    # Construct normalized confusion matrix
    num_cluster = len(np.unique(y_true))
    num_pred = len(np.unique(y_pred))
    NC = np.zeros([num_cluster,num_pred], dtype = int)

    for i, group_true in enumerate(np.unique(y_true)):
        a = y_true == group_true
        a = np.squeeze(a)
        for j, group_pred in enumerate(np.unique(y_pred)):
            b = y_pred == group_pred
            b = np.squeeze(b)
            NC[i,j] = sum(a & b)

    NC = pd.DataFrame(NC, columns = np.unique(y_pred), index = np.unique(y_true))
    
    return NC
