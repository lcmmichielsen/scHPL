# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:16:59 2019

@author: Lieke
"""
import numpy as np
import pandas as pd
import copy as cp
from newick import *

def predict_labels(testdata, tree, offsets = {}):
    '''
    Use an hierarchical classifier to predict the labels of a test set. 
    
    Parameters
    ----------
    testdata: data (cells x genes)
    tree: tree build for the training data using newick.py and trained 
    using build_classifier.py
    offsets: offsets that should be used for each node. This is a dict where
    the key is the name of the node and the value is the offset.
    
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
        
        while len(parentnode.descendants) > 0:
                        
            best_child = None
            max_score = float('-inf')
            
            # Predict if the cell belongs to one of the descendants
            for n in parentnode.descendants:
                label, score = predict_node(testpoint, n, offsets)
                
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
        
        # Label cell with lastly predicted label
        labels_all.append(labels[-1])
        
    return np.asarray(labels_all)
        
def predict_node(testpoint, n, offsets):
    testpoint_transformed = testpoint
    if n.get_dimred():
        pca, pcs = n.get_pca()
        testpoint_transformed = testpoint_transformed[:,pcs]
    
    clf = n.get_classifier()
    label = 0
    
    #linear svm
    if type(clf).__name__ == 'LinearSVC':
        label = clf.predict(testpoint_transformed)
        score = clf.decision_function(testpoint_transformed)
        
    # one-class svm
    else:
        o = offsets[n.name]
        score = clf.score_samples(testpoint_transformed) - o
        if score > 0:
            label = 1
        
    return label, score

