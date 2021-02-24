# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:11:01 2019

@author: Lieke
"""

import pandas as pd
import numpy as np
from .train import train_tree
from .utils import TreeNode, create_tree, print_tree
from .predict import predict_labels
from .update import update_tree

def learn_tree(data, labels, classifier = 'svm_occ', dimred = True, useRE = True, FN = 1, threshold = 0.25, return_missing = True):
    '''
    Apply the hierarchical progressive learning pipeline 
    
    Parameters
    ----------
    data: batch of datasets. The first dimension is the number of datasets.
    Each dataset is cells x genes
    labels: the labels of each dataset
    classifier: the classifier to use, either ('svm' or 'svm_occ')
    dimred: whether to apply dimensionality reduction or not
    threshold: threshold to use during the matching of the labels
    return_missing: If True: return missing nodes to the user, if False: attach
    missing nodes to the root
    
    Return
    ------
    tree_c: trained classification tree
    '''
    
    num_batches = len(data)
    
    data_c = data[0]
    labels_c = labels[0]
    
    tree_c = create_tree('root')
    tree_c = _construct_tree(tree_c, labels_c) 
    
    for i in range(num_batches-1):
        
        print('Iteration ', str(i+1), '\n')
        
        data_2 = data[i+1]
        labels_2 = labels[i+1]
        tree_2 = create_tree('root2')
        tree_2 = _construct_tree(tree_2, labels_2)
        
        # Train the trees
        tree_c = train_tree(data_c, labels_c, tree_c, classifier, dimred, useRE, FN)
        tree_2 = train_tree(data_2, labels_2, tree_2, classifier, dimred, useRE, FN)

        # Predict labels other dataset
        labels_2_pred = predict_labels(data_2, tree_c)
        labels_c_pred = predict_labels(data_c, tree_2)
        
        # Update first tree and labels second dataset
        tree_c, labels_2_new = update_tree(labels_c.values, 
                                           labels_c_pred.reshape(-1,1),
                                           labels_2.values,
                                           labels_2_pred.reshape(-1,1),
                                           threshold, tree_c, 
                                           return_missing = return_missing)
        
        print('\nUpdated tree:')
        print_tree(tree_c)
        
        #concatenate the two datasets
        data_c = pd.DataFrame(np.concatenate((data_c, data_2), axis = 0))
        labels_c = pd.DataFrame(np.concatenate((np.squeeze(labels_c), np.squeeze(labels_2_new)), 
                                axis = 0))
            
    
    
    
    # Train the final tree
    tree_c = train_tree(data_c, labels_c, tree_c, classifier, dimred, useRE, FN)
    
    
    return tree_c
    
    

def _construct_tree(tree, labels):
    '''
    Construct a flat tree
    '''
    
    unique_labels = np.unique(labels)
    
    for ul in unique_labels:
        newnode = TreeNode(ul)
        tree[0].add_descendant(newnode)
        
    return tree