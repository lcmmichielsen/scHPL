# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:16:59 2019

@author: Lieke
"""
import numpy as np
import pandas as pd
from newick import *

def predict_labels(testdata, tree):
    '''
    Use an hierarchical classifier to predict the labels of a test set. 
    
    Parameters
    ----------
    testdata: data (cells x genes)
    tree: classification tree, this tree is created using newick.py and trained 
    using build_classifier.py
    
    Return
    ------
    Predicted labels
    '''

    # Do PCA if needed
    if tree[0].get_dimred():
        pca, pcs = tree[0].get_pca()
        testdata = pca.transform(testdata)
        testdata = pd.DataFrame(testdata)
    
    labels_all = []
    for idx, testpoint in enumerate(testdata.values):
        testpoint = testpoint.reshape(1,-1)
        
        labels = []
        scores = []
        parentnode = tree[0]
        labels.append(tree[0].name)
        scores.append(-1)
        
        # continue until a leaf node is reached
        while len(parentnode.descendants) > 0:
                        
            best_child = None
            max_score = float('-inf')
            
            # Predict if the cell belongs to one of the descendants
            for n in parentnode.descendants:
                label, score = predict_node(testpoint, n)
                
                if (label == 1) & (score > max_score):
                    best_child = n
                    max_score = score
            
            # If so, continue with the best child
            if(best_child != None):
                parentnode = best_child
                scores.append(max_score)
                labels.append(best_child.name)
            # Else, stop
            else:
                break
        
        # Label cell with last predicted label
        labels_all.append(labels[-1])
        
    return np.asarray(labels_all)
        
def predict_node(testpoint, n):
    '''
    Use the local classifier of a node to predict the label of a cell
    
    Parameters
    ----------
    testpoint: data (1 x genes)
    n: node of the tree
    
    Return
    ------
    label: indicates whether the samples is positive (1)
    score: signed distance of a cell to the decision boundary
    
    '''
    
    testpoint_transformed = testpoint
    if n.get_dimred():
        pca, pcs = n.get_pca()
        testpoint_transformed = testpoint_transformed[:,pcs]
    
    clf = n.get_classifier()
    label = clf.predict(testpoint_transformed)
    score = clf.decision_function(testpoint_transformed)
        
    return label, score

