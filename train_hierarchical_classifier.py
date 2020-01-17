# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:11:01 2019

@author: Lieke
"""

import pandas as pd
import numpy as np
import time as tm
from newick import *
from build_classifier import *
from predict_labels import *
from update_tree import *

def train_hierarchical_classifier(data, labels, classifier = 'svm_occ', dimred = True, offset_touse = 'Default', threshold = 0.25):
    '''
    Apply the hierarchical progressive learning pipelin 
    
    Parameters
    ----------
    data: batch of datasets. The first dimension is the number of datasets.
    Each dataset is cells x genes
    labels: the labels of each dataset
    classifier: the classifier to use, either ('svm' or 'svm_occ')
    dimred: whether to apply dimensionality reduction or not
    offset: which offset to use for the one-class SVM ('Default', '1st', 
    '0.5th' or 'Minimum')
    threshold: threshold to use during the matching of the labels
    
    Return
    ------
    tree_c: trained classification tree
    offset_final: offset used
    '''
    
    num_batches = len(data)
    
    data_c = data[0]
    labels_c = labels[0]
    
    tree_c = loads('root')
    tree_c = construct_tree(tree_c, labels_c) 
    
    for i in range(num_batches-1):
        
        data_2 = data[i+1]
        labels_2 = labels[i+1]
        tree_2 = loads('root2')
        tree_2 = construct_tree(tree_2, labels_2)
        
        ### Train the first tree
        scores, names, offset = train_tree(data_c, labels_c, tree_c, 
                                           classifier = classifier, 
                                           dimred = dimred)
        
        offset_c = get_offset(scores,names,offset, offset_touse)
        
        ### Predict labels second dataset
        labels_2_pred = predict_labels(data_2, tree_c, offsets = offset_c)
        
        ### Train the second tree
        scores, names, offset = train_tree(data_2, labels_2, tree_2, 
                                           classifier = classifier, 
                                           dimred = dimred)
        
        
        offset_2 = get_offset(scores,names,offset, offset_touse)
        
        ### Predict labels first dataset
        labels_c_pred = predict_labels(data_c, tree_2, offsets = offset_2)
        
        ### Update first tree and labels second dataset
        tree_c, labels_2_new = update_tree(labels_c.values, 
                                           labels_c_pred.reshape(-1,1),
                                           labels_2.values,
                                           labels_2_pred.reshape(-1,1),
                                           threshold, tree_c)
        
        print('\n\nUpdate tree:')
        print('Root:', tree_c[0])
        for n in tree_c[0].descendants:
            print(n.name)
            for i in n.descendants:
                print(n.name, i.name)
                for j in i.descendants:
                    print(n.name, i.name, j.name)
        
        #concatenate the two datasets
        data_c = pd.DataFrame(np.concatenate((data_c, data_2), axis = 0))
        labels_c = pd.DataFrame(np.concatenate((np.squeeze(labels_c), np.squeeze(labels_2_new)), 
                                axis = 0))
        
        print('\n\nNew labels:', np.unique(labels_c))
    
    
    
    
    ### Train the final tree
    scores, names, offset = train_tree(data_c, labels_c, tree_c, 
                                       classifier = classifier, 
                                       dimred = dimred)
    
    offset_final = get_offset(scores,names,offset, offset_touse)
    
    return tree_c, offset_final
    
    

def construct_tree(tree, labels):
    '''
    Construct a flat tree
    '''
    
    unique_labels = np.unique(labels)
    
    for ul in unique_labels:
        newnode = Node(ul)
        tree[0].add_descendant(newnode)
        
    return tree

def get_offset(scores, names, offset, offset_touse):
    
    if(offset_touse == 'Default'):
        o = {}
        for i in range(len(names)):
            o[names[i]] = offset[i]

    elif(offset_touse == '1st'):
        o = {}
        for i in range(len(names)):
            o[names[i]] = np.percentile(scores[i][0],1)
    
    elif(offset_touse == '0.5th'):
        o = {}
        for i in range(len(names)):
            o[names[i]] = np.quantile(scores[i][0],0.005)

    else:
        o = {}
        for i in range(len(names)):
            o[names[i]] = np.min(scores[i][0])
        
    return o