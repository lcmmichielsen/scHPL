# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:48:26 2019

@author: Lieke
"""

import numpy as np
import pandas as pd
from .utils import TreeNode

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


def confusion_matrix(true_labels, pred_labels):
    '''
    Construct a confusion matrix
    
    Parameters
    ----------
    true_labels: vector with the true labels 
    pred_labels: vector with the predicted labels
        
    Return
    ------
    conf: confusion matrix
    '''
    
    num_cluster = len(np.unique(true_labels))
    num_pred = len(np.unique(pred_labels))
    conf = np.zeros([num_cluster,num_pred], dtype = int)

    for i, group_true in enumerate(np.unique(true_labels)):
        a = true_labels == group_true
        a = np.squeeze(a)
        for j, group_pred in enumerate(np.unique(pred_labels)):
            b = pred_labels == group_pred
            b = np.squeeze(b)
            conf[i,j] = sum(a & b)

    conf = pd.DataFrame(conf, columns = np.unique(pred_labels), index = np.unique(true_labels))
    
    return conf
