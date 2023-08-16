# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:37:16 2019

@author: Lieke
"""

import numpy as np
from numpy import linalg as LA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_ind
from .utils import TreeNode
import copy as cp

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@ignore_warnings(category=ConvergenceWarning)
def train_tree(data, 
               labels, 
               tree: TreeNode, 
               classifier: Literal['knn','svm','svm_occ'] = 'knn', 
               dimred: bool = False, 
               useRE: bool = True, 
               FN: float = 0.5, 
               n_neighbors: int = 50,
               dynamic_neighbors: bool = True,
               distkNN: int = 99):
    '''Train a hierarchical classifier. 
    
        Parameters
        ----------
        data: array_like 
            Training data (cells x genes)
        labels: array_like
            Cell type labels of the training data
        tree: TreeNode 
            Classification tree to train (can be build using utils.create_tree())
        classifier: String = 'knn'
            Classifier to use (either 'svm', 'svm_occ' or 'knn').
        dimred: Boolean = False
            If 'True', PCA is applied before training the classifier.
        useRE: Boolean = True
            If 'True', cells are also rejected based on the reconstruction error.
        FN: Float = 0.5
            Percentage of false negatives allowed when determining the threshold
            for the reconstruction error.
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

        
        Returns
        -------
        Trained classification tree
    '''
    
    numgenes = np.shape(data)[1]
    if numgenes > 100:
        num_components = 100
    else:
        num_components = 0.9
    
    if(useRE == True):
        
        # First determine the threshold for the reconstruction error
        # using an extra cross-validation loop
        perc = 100-(FN)
        
        sss = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
        sss.get_n_splits(data, labels)
        
        RE = []
        
        for trainindex, testindex in sss.split(data, labels):
                
            train = data[trainindex]
            test = data[testindex]
            
            pca = PCA(n_components = num_components, random_state = 0)
            pca.fit(train)
            
            test_t = pca.transform(test)
            test_rec = pca.inverse_transform(test_t)
        
            RE_error2 = LA.norm(test - test_rec, axis = 1)
                    
            RE.append(np.percentile(RE_error2,perc))
        
        pca = PCA(n_components = num_components, random_state = 0)
        pca.fit(data)
        tree[0].set_pca(pca, None) #Save PCA transformation to the root node, so we can apply it to a test set
                        
        tree[0].set_RE(np.median(RE))

    if(dimred == True):
        if(useRE == False):
            pca = PCA(n_components = num_components, random_state = 0)
            pca.fit(data)
            tree[0].set_pca(pca, None) #Save PCA transformation to the root node, so we can apply it to a test set
        
        tree[0].set_dimred(True)
        
        data = pca.transform(data)
    
    # Recursively train the classifiers for each node in the tree
    if(classifier == 'knn'):
        labels_train = cp.deepcopy(labels)
        try: 
            labels_train = labels_train.values
        except:
            None
        _,_ = _train_parentnode(data, labels_train, tree[0], n_neighbors, 
                                dynamic_neighbors, distkNN)
    else:
        for n in tree[0].descendants:
            _ = _train_node(data, labels, n, classifier, dimred, numgenes)

    return tree


def _train_node(data, labels, n, classifier, dimred, numgenes):
    '''Train a linear of one-class SVM (binary classifier) for each node.
    
        Parameters
        ----------
        data: training data
        labels: labels of the training data
        n: node to train the classifier for
        classifier: which classifier to use
        dimred: dimensionality reduction
        numgenes: number of genes in the training data
        
        Returns
        -------
        group: vector which indicates the positive samples of a node
    '''
    
    group = np.zeros(len(labels), dtype = int)

    if n.is_leaf:
        group[np.isin(labels, n.name)] = 1
    else:
        if n.name != None:
            group[np.isin(labels, n.name)] = 1
        for j in n.descendants:
            group_new = _train_node(data, labels, j, classifier, dimred, numgenes)
            group[np.where(group_new == 1)[0]] = 1
            
    if(dimred):
        data = _find_pcs(data, labels, group, n, numgenes)
    
    if(classifier == 'svm'):
        _train_svm(data, labels, group, n)
    else:
        _train_occ(data, labels, group, n)
        
    return group

def _train_parentnode(data, labels, n, n_neighbors, dynamic_neighbors, distkNN):
    '''Train a knn classifier. In contrast to the linear svm and oc svm, this 
        is trained for each parent node instead of each child node
        
        Parameters
        ----------
        data: training data
        labels: labels of the training data
        n: node to train the classifier for
        classifier: which classifier to use
        dimred: dimensionality reduction
        numgenes: number of genes in the training data
        
        Return
        ------
        group: vector which indicates the positive samples of a node
    '''
    
    group = np.zeros(len(labels), dtype = int)

    if n.is_leaf:
        group[np.squeeze(np.isin(labels, n.name))] = 1
        labels[np.squeeze(np.isin(labels, n.name))] = n.name[0]
        return group, labels
    else:
        for j in n.descendants:
            group_new, labels_new = _train_parentnode(data, labels, j, 
                                                      n_neighbors, dynamic_neighbors,
                                                      distkNN)
            group[np.where(group_new == 1)[0]] = 1
            labels[np.where(group_new == 1)[0]] = labels_new[np.where(group_new == 1)[0]]
        if n.name != None:
            # special case; if n has only 1 child
            if len(n.descendants) == 1:
                group[np.squeeze(np.isin(labels, n.name))] = 1
            # train_knn 
            _train_knn(data,labels,group,n,n_neighbors,dynamic_neighbors,distkNN)
            # rename all group == 1 to node.name
            group[np.squeeze(np.isin(labels, n.name))] = 1
            labels[group==1] = n.name[0]
                    
    return group, labels



def _find_pcs(data, labels, group, n, numgenes):
    '''Do a ttest between the positive and negative samples to find explaining pcs.'''
    
    group = _find_negativesamples(labels, group, n)
    
    # positive samples
    this_class = np.where(group == 1)[0]
    this_data = data[this_class]
    
    # negative samples
    other_class = np.where(group == 2)[0]
    other_data = data[other_class]
    
    statistic, pvalue = ttest_ind(this_data, other_data, equal_var = False)

    explaining_pcs = np.where(pvalue*numgenes <= 0.05)[0]  
    
    ## If there are no explaining PCs, just pick the 5 top ones
    if len(explaining_pcs) == 0:
        explaining_pcs = np.argsort(pvalue)[:5]
            
    data = data[:,explaining_pcs]
    
    # Save the explaining pcs in the tree
    n.set_pca(None, explaining_pcs)
    
    return data


def _train_svm(data, labels, group, n):
    '''Train a linear svm and attach to the node
    
        Parameters:
        -----------
        data: training data
        group: indicating which cells of the training data belong to that node
        n: node
    '''
    
    # group == 1 --> positive samples
    # group == 2 --> negative samples
    group = _find_negativesamples(labels, group, n) 
    idx_svm = np.where((group == 1) | (group == 2))[0]
    data_svm = data[idx_svm]
    group_svm = group[idx_svm]
    
    clf = svm.LinearSVC(random_state=1).fit(data_svm, group_svm)
    
    n.set_classifier(clf) #save classifier to the node
    

def _train_knn(data, labels, group, n, n_neighbors, dynamic_neighbors, distkNN):
    '''Train a linear svm and attach to the node
    
        Parameters:
        -----------
        data: training data
        group: indicating which cells of the training data belong to that node
        n: node
    '''

    idx_knn = np.where(group == 1)[0]
    data_knn = data[idx_knn]
    labels_knn = np.squeeze(labels[idx_knn])
    
    k = int(n_neighbors)
    
    if dynamic_neighbors:
        smallest_pop = np.min(np.unique(labels_knn, return_counts=True)[1])
        if smallest_pop < n_neighbors:
            k = int(smallest_pop)
    
    #print(k)
    
    # print(np.unique(labels_knn))
    
    # Check if FAISS is installed, if not use sklearn KNN
    try:
        import faiss
        from .faissKNeighbors import FaissKNeighbors 
        clf = FaissKNeighbors(k=k)
        clf.fit(data_knn, labels_knn)
        #print('Using FAISS library')

    except:
        #print('Using KNN')
        clf = KNN(weights='distance',
                  n_neighbors=k,
                  metric='euclidean').fit(data_knn, labels_knn)
    
    if((n.name[0] == 'root') | (n.name[0] == 'root2')):
        dist,idx = clf.kneighbors(data, return_distance=True)
        n.set_maxDist(np.percentile(np.mean(dist[:,1:],axis=1),distkNN))
    
    n.set_classifier(clf) #save classifier to the node

        

def _train_occ(data, labels, group, n):
    '''Train a one-class-classifier SVM and attach to the node
    
        Parameters:
        ----------
        data: training data
        group: indicating which cells of the training data belong to that node
        n: node
    '''
    
    data_group = data[np.where(group == 1)[0]]
    
    clf = svm.OneClassSVM(gamma = 'scale', nu = 0.05).fit(data_group)
    n.set_classifier(clf) #save classifier to the node
    
    return 

def _find_negativesamples(labels, group, n):
    
    a = n.ancestor
    
    # Find the 'sister' nodes
    for i in a.descendants:
        if(i.name != n.name):
            for j in i.walk():
                group[np.isin(labels, j.name)] = 2
                
    # If we find no sisters, we compare with the other samples of its ancestor
    if(len(np.where(group == 2)[0])) == 0:
        group[np.isin(labels, a.name)] = 2

    return group

