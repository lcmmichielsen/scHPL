# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:37:16 2019

@author: Lieke
"""

from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import time as tm
from newick import *


def train_tree(data, labels, tree, classifier = 'svm_occ', dimred = True):
    '''
    Train the hierarchical classifier. 
    
    Parameters
    ----------
    data: training data (cells x genes)
    labels: labels of the training data
    tree: tree build for the training data using newick.py
    classifier: which classifier to use ('svm' or 'svm_occ')
    
    
    Return
    ------
    Score: the scores of the positive and negative samples for each node
    (these scores can be used to find the minimum, 1st perc., and 0.5th perc.
    offset)
    Offset: the default offset for each node
    Name: name of each node (same order as the scores, so the scores can be 
    linked to a cell populations)
    '''
    
    score  = []
    name = []
    offset = []
    numgenes = np.shape(data)[1]
    
    if(dimred == True):
        pca = PCA(n_components = 0.9)
        pca.fit(data)
        tree[0].set_pca(pca, None) #Save PCA transformation to the root node, so we can apply it to a test set
        data = pca.transform(data)
        data = pd.DataFrame(data)
    
    #recursively train the tree
    for n in tree[0].descendants:
        group = train_classifier(data, labels, n, classifier, dimred, score, name, offset, numgenes)

    return score, name, offset


def train_classifier(data, labels, n, classifier, dimred, score, name, offset, numgenes):
    
    scores = 0
    o = 0
    
    if n.is_leaf:
        group = np.zeros(len(labels), dtype = int)
        group[np.where(labels == n.name)[0]] = 1
    else:
        group = np.zeros(len(labels), dtype = int)
        if n.name != None:
            group[np.where(labels == n.name)[0]] = 1
        for j in n.descendants:
            group_new = train_classifier(data, labels, j, classifier, dimred, score, name, offset, numgenes)
            group[np.where(group_new == 1)[0]] = 1
            
    if(dimred):
        data = apply_pca(data, labels, group, n, numgenes)
    
    if(classifier == 'svm'):
        train_svm(data, labels, group, n)
    else:
        scores, o = train_occ(data, labels, group, n)
    
    score.append(scores)
    name.append(n.name)
    offset.append(o)
    
    return group


def apply_pca(data, labels, group, n, numgenes):
    '''
    Apply pca to the data and afterwards do a ttest to find the pcs that explain 
    the variance between the node and the rest.
    '''
    
    
    group = find_sisternodes(labels, group, n)
    
    # positive samples
    this_class = np.where(group == 1)[0]
    this_data = data.iloc[this_class]
    
    # negative samples
    other_class = np.where(group == 2)[0]
    other_data = data.iloc[other_class]
    
    statistic, pvalue = ttest_ind(this_data, other_data, equal_var = False)

    explaining_pcs = np.where(pvalue*numgenes <= 0.05)[0]  
    
    ## If there are no explaining PCs, just pick the 5 top ones
    if len(explaining_pcs) == 0:
        explaining_pcs = np.argsort(pvalue)[:5]
    
    data = data.iloc[:,explaining_pcs]
    
    n.set_pca(None, explaining_pcs)
    
    return data


def train_svm(data, labels, group, n):
    '''
    Train a linear svm and attach to the node
    
    Parameters:
    -----------
    data: training data
    group: indicating which cells of the training data belong to that node
    n: node
    '''
    
    group = find_sisternodes(labels, group, n) 

    # group == 1 --> positive samples
    # group == 2 --> negative samples
    idx_svm = np.where((group == 1) | (group == 2))[0]
    data_svm = data.iloc[idx_svm]
    group_svm = group[idx_svm]

    clf = svm.LinearSVC().fit(data_svm, group_svm)
    n.set_classifier(clf) #save classifier to the node
    
        

def train_occ(data, labels, group, n):
    '''
    Train a one-class-classifier SVM and attach to the node
    
    Parameters:
    ----------
    data: training data
    group: indicating which cells of the training data belong to that node
    n: node
    '''
    
    scores = []
    group = find_sisternodes(labels, group, n)           

    # group == 1 --> positive samples
    # group == 2 --> negative samples
    data_group = data.iloc[np.where(group == 1)[0]]
    data_other = data.iloc[np.where(group == 2)[0]]
    clf = svm.OneClassSVM(gamma = 'scale', nu = 0.05).fit(data_group)
    scores_group = clf.score_samples(data_group)
    scores_other = clf.score_samples(data_other)
    scores.append(scores_group)
    scores.append(scores_other)
    off = clf.offset_
    n.set_classifier(clf) #save classifier to the node
    
    return scores, off

def find_sisternodes(labels, group, n):
    
    a = n.ancestor
    
    # Find the 'sister' nodes
    for i in a.descendants:
        if(i.name != n.name):
            for j in i.walk():
                group[np.where(labels == j.name)[0]] = 2
                
    # If we find no sisters, we compare with the other samples of its ancestor?
    if(len(np.where(group == 2)[0])) == 0:
        group[np.where(labels == a.name)[0]] = 2

    return group









