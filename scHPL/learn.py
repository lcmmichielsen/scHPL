# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:11:01 2019

@author: Lieke
"""

import numpy as np
from anndata import AnnData

from .train import train_tree
from .utils import TreeNode, create_tree, print_tree
from .predict import predict_labels
from .update import update_tree
# from train import train_tree
# from utils import TreeNode, create_tree, print_tree
# from predict import predict_labels
# from update import update_tree

try:
    from typing import Literal, Optional
except ImportError:
    from typing_extensions import Literal, Optional


def learn_tree(data: AnnData,
               batch_key: str,
               batch_order: list,
               cell_type_key: str,
               tree: TreeNode = None,
               retrain: bool = False,
               batch_added: list = None,
               classifier: Literal['knn','svm','svm_occ'] = 'knn',
               n_neighbors: int = 50,
               dynamic_neighbors: bool = True,
               distkNN: int = 99,
               dimred: bool = False,
               useRE: bool = True,
               FN: float = 0.5,
               rej_threshold: float = 0.5,
               match_threshold: float = 0.25,
               attach_missing: bool = False,
               print_conf: bool = False,
               gpu: Optional[int] = None,
               compress: bool = False
):
    
    '''Learn a classification tree based on multiple labeled datasets.
        
        Parameters
        ----------
        data: AnnData 
            AnnData matrix containing aligned datasets.
        batch_key: String
            Column name in adata.obs containing batch information.
        batch_order: List
            List containing the order in which the batches should be added 
            to the tree.
        cell_type_key: String
            Column name in adata.obs containing the celltype labels.
        tree: TreeNode = None
            Existing tree to update with the new datasets. 
        retrain: Boolean = False
            If 'True', the inputted tree will be retrained (needed if tree or 
            datasets are changed after intial construction).
        batch_added: List = None
            List that indicates which batches were used to build the existing tree.
        classifier: String = 'knn'
            Classifier to use (either 'svm', 'svm_occ' or 'knn').
        n_neighbors: int = 50
            Number of neighbors for the kNN classifier (only used when 
            classifier='knn').
        dynamic_neighbors: bool = True
            Number of neighbors for the kNN classifier can change when a node 
            contains a very small cell population. k is set to 
            min(n_neighbors, smallest-cell-population)
        distkNN: int = 99
            Used to determine the threshold for the maximum distance between a 
            cell and it's closest neighbor of the training set. Threshold is 
            set to the distkNN's percentile of distances within the training
            set
        dimred: Boolean = False
            If 'True', PCA is applied before training the classifier.
        useRE: Boolean = True
            If 'True', cells are also rejected based on the reconstruction error.
        FN: Float = 0.5
            Percentage of false negatives allowed when determining the threshold
            for the reconstruction error.
        rej_threshold: Float = 0.5
            If prediction probability lower that this threshold, a cell is rejected.
            (only used when using kNN classifier)
        match_threshold: Float = 0.25
            Threshold to use when matching the labels.
        attach_missing: Boolean = False
            If 'True' missing nodes are attached to the root node.
        print_conf: Boolean = False
            Whether to print the confusion matrices during the matching step.
        gpu: int = None
            GPU index to use for the Faiss library (only used when classifier='knn')
        compress: Boolean = False
            If 'True', the Faiss index is compressed (only used when classifier='knn')
            
        Returns
        -------
        Trained classification tree and a list with the missing populations.
    '''
    
    missing_pop=[]
    
    xx = data.X
    labels = np.array(data.obs[cell_type_key].values, dtype=str)
    batches = data.obs[batch_key]
    
    # If no existing tree, construct tree for first batch    
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
    
    print('Starting tree:')
    print_tree(tree)
    
    for b in batch_order:
        
        print('\nAdding dataset', str(b), 'to the tree')
        
        idx_2 = np.where(batches == b)[0]
        data_2 = xx[idx_2]
        labels_2 = labels[idx_2]
        tree_2 = create_tree('root2')
        tree_2 = _construct_tree(tree_2, labels_2)
                
        # Train the trees
        if retrain:
            tree = train_tree(data_1, labels_1, tree, classifier, 
                              dimred, useRE, FN, n_neighbors, dynamic_neighbors,
                              distkNN, gpu=gpu, compress=compress)
        else:
            retrain = True 
        
        tree_2 = train_tree(data_2, labels_2, tree_2, classifier, 
                            dimred, useRE, FN, n_neighbors, dynamic_neighbors,
                            distkNN, gpu=gpu, compress=compress)
        
        # Predict labels other dataset
        labels_2_pred,_ = predict_labels(data_2, tree, threshold=rej_threshold)
        labels_1_pred,_ = predict_labels(data_1, tree_2, threshold=rej_threshold)
        
        # Update first tree and labels second dataset
        tree, mis_pop = update_tree(tree, labels_1.reshape(-1,1),
                                    labels_1_pred.reshape(-1,1),
                                    labels_2.reshape(-1,1),
                                    labels_2_pred.reshape(-1,1),
                                    match_threshold, 
                                    attach_missing,
                                    print_conf)
        
        missing_pop.extend(mis_pop)
        
        print('\nUpdated tree:')
        print_tree(tree, np.unique(labels_2))
        
        #concatenate the two datasets
        data_1 = np.concatenate((data_1, data_2), axis = 0)
        labels_1 = np.concatenate((np.squeeze(labels_1), np.squeeze(labels_2)), 
                                axis = 0)
    
    # Train the final tree
    tree = train_tree(data_1, labels_1, tree, classifier, dimred, useRE, FN,
                      n_neighbors, dynamic_neighbors, distkNN)
    
    return tree, missing_pop
    
    

def _construct_tree(tree, labels):
    '''
    Construct a flat tree.
    '''
    
    unique_labels = np.unique(labels)
    
    for ul in unique_labels:
        newnode = TreeNode([ul])
        tree[0].add_descendant(newnode)
        
    return tree