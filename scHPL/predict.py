# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:16:59 2019

@author: Lieke
"""
import numpy as np
from numpy import linalg as LA
from .utils import TreeNode

def predict_labels(testdata, 
                   tree: TreeNode, 
                   threshold: float = 0.5):
    '''Use the trained tree to predict the labels of a new dataset. 
    
        Parameters
        ----------
        testdata: array_like 
            Data to classify (cells x genes)
        tree: TreeNode
            Trained classification tree
        threshold: Float = 0.5
            If prediction probability lower that this threshold, a cell is rejected.
            (only used when using kNN classifier)
        
        Returns
        ------
        Predicted labels
    '''
    
    useRE = False
    # First reject cells based on reconstruction error
    if tree[0].get_RE() > 0:
        useRE = True
        
        t = tree[0].get_RE()
        
        pca, pcs = tree[0].get_pca()
        test_t = pca.transform(testdata)
        test_rec = pca.inverse_transform(test_t)
        
        RE_error2 = LA.norm(testdata - test_rec, axis = 1)
        rej_RE = RE_error2 > t

    # Do PCA if needed
    dimred = False
    if tree[0].get_dimred():
        pca, pcs = tree[0].get_pca()
        testdata = pca.transform(testdata)
        dimred = True
    
    labels_all = []
    for idx, testpoint in enumerate(testdata):
        # print(idx)
        if useRE:   
            if rej_RE[idx]:
                labels_all.append('Rejected (RE)')
                # labels_all.append(tree[0].name[0])
                continue
        
        testpoint = testpoint.reshape(1,-1)
        
        labels = []
        scores = []
        parentnode = tree[0]
        labels.append(tree[0].name[0])
        scores.append(-1)
        
        #### If we have trained knn
        if parentnode.classifier:
            
            ### Reject cells based on distance
            predict=True
            dist,idx = parentnode.classifier.kneighbors(testpoint, return_distance=True)
            if(np.mean(dist) > parentnode.get_maxDist()):
                labels.append('Rejection (dist)')
                predict=False
            
            while (parentnode.classifier != None) & predict:
                # print(parentnode.name)
                label, score = _predict_node(testpoint, parentnode, dimred)
                
                #If score higher than threshold -> iterate further over tree
                if score > threshold:
                    labels.append(label[0])
                    oldparent = parentnode
                    for n in parentnode.descendants:
                        if n.name[0] == label:
                            parentnode = n
                    if parentnode.name == oldparent.name:
                        break
                else:
                    break
                
        
        
        #### If we have trained svm or svm oc
        # continue until a leaf node is reached
        else:
            while len(parentnode.descendants) > 0:
                            
                best_child = None
                max_score = float('-inf')
                
                # Predict if the cell belongs to one of the descendants
                for n in parentnode.descendants:
                    label, score = _predict_node(testpoint, n, dimred)
                    
                    if (label == 1) & (score > max_score):
                        best_child = n
                        max_score = score
                
                # If so, continue with the best child
                if(best_child != None):
                    parentnode = best_child
                    scores.append(max_score)
                    labels.append(best_child.name[0])
                # Else, stop
                else:
                    break
        
        # Label cell with last predicted label
        labels_all.append(labels[-1])
        
    return np.asarray(labels_all)
        
def _predict_node(testpoint, n, dimred):
    '''Use the local classifier of a node to predict the label of a cell.
    
    Parameters
    ----------
    testpoint: data (1 x genes)
    n: node of the tree
    
    Returns
    -------
    label: indicates whether the samples is positive (1)
    score: signed distance of a cell to the decision boundary
    '''
    
    testpoint_transformed = testpoint
    if dimred:
        pca, pcs = n.get_pca()
        if np.any(pcs):
            testpoint_transformed = testpoint_transformed[:,pcs]
    
    clf = n.get_classifier()
    label = clf.predict(testpoint_transformed)
    try:
        score = clf.decision_function(testpoint_transformed)
    except:
        score = clf.predict_proba(testpoint_transformed)
        score = np.max(score)
        
    return label, score

