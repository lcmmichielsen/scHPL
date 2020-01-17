# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:48:26 2019

@author: Lieke
"""

import numpy as np
from newick import *

def hierarchical_F1(true_labels, pred_labels, tree):
    
    sum_p = 0
    sum_t = 0
    sum_o = 0
        
        
    for i in range(len(true_labels)):
        true_lab = true_labels[i]
        pred_lab = pred_labels[i]
        
        found = 0
        
        set_true = []
        set_pred = []
        
        for n in tree[0].walk('postorder'):
            if(n.name == true_lab):
                found += 1
                set_true.append(n.name)
                a = n.ancestor
                while(a != None):
                    set_true.append(a.name)
                    a = a.ancestor
                
                if found == 2:
                    break
                    
            if(n.name == pred_lab):
                found += 1
                set_pred.append(n.name)
                a = n.ancestor
                while(a != None):
                    if(a.name == true_lab):
                        set_pred = []
                    set_pred.append(a.name)
                    a = a.ancestor

                if found == 2:
                    break
        
        common = len(np.intersect1d(set_pred, set_true))
        pred_len = len(set_pred)
        true_len = len(set_true)
        
        
        sum_p += pred_len - 1 # -1 to remove root
        sum_t += true_len - 1 # -1 to remove root
        sum_o += common - 1 # -1 to remove root

    hP = sum_o/sum_p
    hR = sum_o/sum_t      
                
    hF1 = (2 * hP * hR)/(hP + hR)
    
    
    return hF1