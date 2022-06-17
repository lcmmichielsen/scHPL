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

def hierarchical_F1(true_labels, 
                    pred_labels, 
                    tree: TreeNode):
    '''Calculate the hierarchical F1-score
    
        Parameters
        ----------
        true_labels: array_like
            True labels 
        pred_labels: array_like
            Predicted labels
        tree: TreeNode 
            Classification tree used to predict the labels
            
        Returns
        -------
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
            if(np.isin(true_lab, n.name)):
                found += 1
                set_true.append(n.name[0])
                a = n.ancestor
                while(a != None):
                    set_true.append(a.name[0])
                    a = a.ancestor
                
                if found == 2:
                    break
                    
            if(np.isin(pred_lab, n.name)):
                found += 1
                set_pred.append(n.name[0])
                a = n.ancestor
                while(a != None):
                    if(np.isin(true_lab, a.name)):
                        set_pred = []
                    set_pred.append(a.name[0])
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
    '''Construct a confusion matrix.
    
        Parameters
        ----------
        true_labels: array_like 
            True labels of the dataset
        pred_labels: array_like
            Predicted labels
            
        Returns
        -------
        conf: confusion matrix
    '''
    
    true_labels = pd.DataFrame(true_labels).reset_index(drop=True)
    pred_labels = pd.DataFrame(pred_labels).reset_index(drop=True)
    yall = pd.concat([true_labels, pred_labels], axis=1)
    yall.columns = ['ytrue', 'ypred']
    conf = pd.crosstab(yall['ytrue'], yall['ypred'])

    return conf

def heatmap(true_labels, 
            pred_labels, 
            order_rows: list = None, 
            order_cols: list = None, 
            transpose: bool = False, 
            cmap: str = 'Reds', 
            title: str = None, 
            annot: bool = False,
            xlabel: str = 'Predicted labels', 
            ylabel: str = 'True labels', 
            shape = (10,10), 
            **kwargs):
    '''Plot a confusion matrix as a heatmap.
    
        Parameters
        ----------
        true_labels: array_like
            True labels of the dataset
        pred_labels: array_like
            Predicted labels
        order_rows: List = None
            Order of the cell types (rows)
        order_cols: List = None
            Order of the cell types (cols)
        transpose: Boolean = False
            If True, the rows become the true labels instead of the columns.
        cmap : String = 'reds'
            Colormap to use. Can be any matplotlib colormap
        title : String = None
            Title of the plot.
        annot : Boolean = False
            If true, the data value is added to each cell. 
        xlabel : String = 'Predicted labels'
            Text of the x label
        ylabel : String = 'True labels'
            Text of the y label
        shape : (float, float) = (10,10)
            Size of the plot
        **kwargs : 
            Other keyword args for sns.heatmap().

    Returns
    -------
    None.

    '''

    #Get confusion matrix & normalize
    conf = confusion_matrix(true_labels, pred_labels) 

    if transpose:
        conf = np.transpose(conf)

    conf2 = np.divide(conf,np.sum(conf.values, axis = 1, keepdims=True))   

    if order_rows is None:
        num_rows = np.shape(conf2)[0]
        order_rows = np.linspace(0, num_rows-1, num=num_rows, dtype=int)
        order_rows = np.asarray(conf2.index)
    else:
        xx = np.setdiff1d(order_rows, conf2.index)
        test = pd.DataFrame(np.zeros((len(xx), np.shape(conf2)[1])), index = xx, columns=conf2.columns)
        conf2 = pd.concat([conf2,test], axis=0)    
    
    if order_cols is None:
        num_cols = np.shape(conf2)[1]
        order_cols = np.linspace(0, num_cols-1, num=num_cols, dtype=int)
        order_cols = np.asarray(conf2.columns)
    else:
        xx = np.setdiff1d(order_cols, conf2.columns)
        test = pd.DataFrame(np.zeros((np.shape(conf2)[0], len(xx))), index = conf2.index, columns=xx)
        conf2 = pd.concat([conf2,test], axis=1)    
    
    plt.figure(figsize=shape)
    if annot:
        sns.heatmap(conf2.loc[order_rows,order_cols], vmin = 0, vmax = 1, 
                cbar_kws={'label': 'Fraction'}, cmap=cmap, 
                annot=conf.iloc[order_rows, order_cols], **kwargs)
    else:
        sns.heatmap(conf2.loc[order_rows,order_cols], vmin = 0, vmax = 1, 
                cbar_kws={'label': 'Fraction'}, cmap=cmap, **kwargs)
    
    if title is not None:
        plt.title(title)
        
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize = 14)
    
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize = 14)
    
#     plt.show()
    
    return plt