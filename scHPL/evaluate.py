# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:48:26 2019

@author: Lieke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    true_labels = pd.DataFrame(true_labels).reset_index(drop=True)
    pred_labels = pd.DataFrame(pred_labels).reset_index(drop=True)
    yall = pd.concat([true_labels, pred_labels], axis=1)
    yall.columns = ['ytrue', 'ypred']
    conf = pd.crosstab(yall['ytrue'], yall['ypred'])

    return conf

def heatmap(true_labels, pred_labels, order_rows = None, order_cols = None, 
            transpose = False, cmap = 'Reds', title = None, annot=False,
            xlabel = 'Predicted labels', ylabel = 'True labels', 
            shape = (10,10), **kwargs):

    #Get confusion matrix & normalize
    conf = confusion_matrix(true_labels, pred_labels) 

    if transpose:
        conf = np.transpose(conf)

    conf2 = np.divide(conf,np.sum(conf.values, axis = 1, keepdims=True))   

    if order_rows is None:
        num_rows = np.shape(conf2)[0]
        order_rows = np.linspace(0, num_rows-1, num=num_rows, dtype=int)
    
    if order_cols is None:
        num_cols = np.shape(conf2)[1]
        order_cols = np.linspace(0, num_cols-1, num=num_cols, dtype=int)
    
    plt.figure(figsize=shape)
    if annot:
        sns.heatmap(conf2.iloc[order_rows,order_cols], vmin = 0, vmax = 1, 
                cbar_kws={'label': 'Fraction'}, cmap=cmap, 
                annot=conf.iloc[order_rows, order_cols], **kwargs)
    else:
        sns.heatmap(conf2.iloc[order_rows,order_cols], vmin = 0, vmax = 1, 
                cbar_kws={'label': 'Fraction'}, cmap=cmap, **kwargs)
    
    if title is not None:
        plt.title(title)
        
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize = 14)
    
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize = 14)
    
    plt.show()
    
