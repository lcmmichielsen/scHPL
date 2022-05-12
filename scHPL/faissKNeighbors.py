# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:57:39 2022

@author: lcmmichielsen
"""

import faiss
import numpy as np

class FaissKNeighbors:
    def __init__(self, k=50):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.unique(x)[np.argmax(np.unique(x, return_counts=True)[1])] for x in votes])
        
        return predictions
    
    def predict_proba(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        probs = np.array([np.unique(x, return_counts=True)[1]/self.k for x in votes])
        
        return probs
        
    def kneighbors(self, X, return_distance=True):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        if return_distance:
            return distances, indices
        else:
            return indices