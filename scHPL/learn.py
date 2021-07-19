# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:11:01 2019

@author: Lieke
"""

import numpy as np

import anndata

from .train import train_tree
from .utils import TreeNode, create_tree, print_tree
from .predict import predict_labels
from .update import update_tree

def learn_tree(data: anndata,
               batch_key: str,
               batch_order: list,
               cell_type_key: str,
               tree: TreeNode = None,
               retrain: bool = False,
               batch_added: list = None,
               classifier: str = 'svm_occ',
               dimred: bool = True,
               useRE: bool = True,
               FN: float = 1,
               threshold: float = 0.25,
               return_missing: bool = True
):
    
    '''
    Apply the hierarchical progressive learning pipeline 
    
    Parameters
    ----------
    data: Anndata 
        anndata object
    batch_key: String
        Key where the batches in the data can be found.
    batch_order: List
        List containing the order in which the batches should be added 
        to the tree.
    cell_type_key: String
        Key where the celltype labels in the data can be found.
    tree: TreeNode = None
        Tree to start updating. 
    retrain: Boolean = False
        If 'True', the inputted tree will be retrained (needed if tree or 
        datasets are changed after intial construction).
    batch_added: List = None
        Indicates which batches were used to build the existing tree.
    classifier: String = 'svm_occ'
        Classifier to use (either 'svm' or 'svm_occ').
    dimred: Boolean = True
        If 'True' PCA is applied before training the classifier.
    useRE: Boolean = True
        If 'True', cells are also rejected based on the reconstruction error.
    FN: Float = 1
        Percentage of false negatives allowed when determining the threshold
        for the reconstruction error.
    threshold: Float = 0.25
        Threshold to use when matching the labels.
    return_missing: Boolean = True
        If 'True' missing nodes are returned to the user, else missing
        nodes are attached to the root node.
        
    Return
    ------
    tree_c: trained classification tree
    '''
    
    missing_pop=[]
    
    xx = data.X
    labels = np.array(data.obs[cell_type_key].values, dtype=str)
    batches = data.obs[batch_key]
        
    if(tree == None):
        tree = create_tree('root')
        firstbatch = batch_order[0]
        batch_order = batch_order[1:]
        idx_1 = np.where(batches == firstbatch)[0]
        labels_1 = labels[idx_1]
        tree = _construct_tree(tree, labels_1)
        retrain = True
    else:
        idx_1 = np.isin(batches, batch_added)
    
    labels_1 = labels[idx_1]
    data_1 = xx[idx_1]
    
    for b in batch_order:
        
        print('Adding dataset', str(b), 'to the tree')
        
        idx_2 = np.where(batches == b)[0]
        data_2 = xx[idx_2]
        labels_2 = labels[idx_2]
        tree_2 = create_tree('root2')
        tree_2 = _construct_tree(tree_2, labels_2)
                
        # Train the trees
        if retrain:
            tree = train_tree(data_1, labels_1, tree, classifier, dimred, useRE, FN)
        else:
            retrain = True 
        
        tree_2 = train_tree(data_2, labels_2, tree_2, classifier, dimred, useRE, FN)
        
        # Predict labels other dataset
        labels_2_pred = predict_labels(data_2, tree)
        labels_1_pred = predict_labels(data_1, tree_2)
        
        # Update first tree and labels second dataset
        tree, labels_2_new, mis_pop = update_tree(labels_1.reshape(-1,1), 
                                           labels_1_pred.reshape(-1,1),
                                           labels_2.reshape(-1,1),
                                           labels_2_pred.reshape(-1,1),
                                           threshold, tree, 
                                           return_missing = return_missing)
        missing_pop.extend(mis_pop)
        
        print('Updated tree:')
        print_tree(tree)
        
        #concatenate the two datasets
        data_1 = np.concatenate((data_1, data_2), axis = 0)
        labels_1 = np.concatenate((np.squeeze(labels_1), np.squeeze(labels_2_new)), 
                                axis = 0)
            
    
    
    
    # Train the final tree
    tree = train_tree(data_1, labels_1, tree, classifier, dimred, useRE, FN)
    
    if return_missing:
        return tree, missing_pop
    else:
        return tree
    
    

def _construct_tree(tree, labels):
    '''
    Construct a flat tree
    '''
    
    unique_labels = np.unique(labels)
    
    for ul in unique_labels:
        newnode = TreeNode(ul)
        tree[0].add_descendant(newnode)
        
    return tree