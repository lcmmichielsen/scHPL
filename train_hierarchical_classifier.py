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

def train_hierarchical_classifier(data, labels, classifier = 'svm_occ', dimred = True, threshold = 0.25):
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
    
    Return
    ------
    tree_c: trained classification tree
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
        
        # Train the trees
        tree_c = train_tree(data_c, labels_c, tree_c, classifier, dimred)
        tree_2 = train_tree(data_2, labels_2, tree_2, classifier, dimred)

        # Predict labels other dataset
        labels_2_pred = predict_labels(data_2, tree_c)
        labels_c_pred = predict_labels(data_c, tree_2)
        
        # Update first tree and labels second dataset
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
            
    
    
    
    # Train the final tree
    tree_c = train_tree(data_c, labels_c, tree_c, classifier, dimred)
    
    
    return tree_c
    
    

def construct_tree(tree, labels):
    '''
    Construct a flat tree
    '''
    
    unique_labels = np.unique(labels)
    
    for ul in unique_labels:
        newnode = Node(ul)
        tree[0].add_descendant(newnode)
        
    return tree