# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:58:14 2019

@author: Lieke
"""

import numpy as np
import pandas as pd
from newick import *

def update_tree(y_true1, y_pred1, y_true2, y_pred2, threshold, tree):
    '''
    Match the labels of two datasets 
    
    Parameters
    ----------
    y_true1: true labels of dataset 1 
    y_pred1: predicted labels of dataset 1 (predicted by tree2)
    y_true2: true labels of dataset 2 
    y_pred2: predicted labels of dataset 2 (predicted by tree1)
    threshold: threshold used to create the binary confusion matrix
    tree: tree belonging to dataset 1
    
    Return
    ------
    tree: the updated tree
    y_true2: the updated labels of y_true2 so that they correspond
             to the labels of y_true1 in case of a perfect match
    '''
        
    # Construct binary confusion matrices
    BC1 = confusion_binary(y_true1, y_pred1, threshold)
    BC2 = confusion_binary(y_true2, y_pred2, threshold)
    
    # Remove non-leaf nodes from the rows of the first matrix
    BC1 = BC1.reindex(index = tree[0].get_leaf_names(), fill_value = False)
    
    # Add the binary matrices
    X = BC2.add(BC1.T, fill_value = 0)
    
    # Find matches
    y_true2 = match_trees(X, tree, y_true2)
    
        
    return tree, y_true2
    
def confusion_binary(y_true, y_pred, threshold):
    '''
    Construct a binary confusion matrix
    '''
    
    # Construct normalized confusion matrix
    num_cluster = len(np.unique(y_true))
    num_pred = len(np.unique(y_pred))
    NC = np.zeros([num_cluster,num_pred])

    for i, group_true in enumerate(np.unique(y_true)):
        a = y_true == group_true
        a = np.squeeze(a)
        for j, group_pred in enumerate(np.unique(y_pred)):
            b = y_pred == group_pred
            b = np.squeeze(b)
            NC[i,j] = sum(a & b)/ sum(a)

    NC = pd.DataFrame(NC, columns = np.unique(y_pred), index = np.unique(y_true))
#    print(NC.values)
    
    
    # Convert NC to BC
    BC = NC > 1
    
    # Find max value and put a True there
    max_values = np.argmax(NC.values, axis = 1)
    for idx, mv in enumerate(max_values):
        BC.iloc[idx,mv] = True
        
    # Next, we look if the difference between the second highest and highest
    # is smaller than the threshold. (and third highest than the second etc.)
    nv = np.ones([np.shape(NC)[0],], dtype = bool) 
    for i in range((np.shape(NC)[1] - 2)):
        sort1 = np.sort(NC.values, axis = 1)[:,-(i+1)]
        sort2 = np.sort(NC.values, axis = 1)[:,-(i+2)]
        new = sort1 - sort2
        nv = nv & (new < threshold) & (sort2 > 0.1)
        if np.sum(nv) > 0:
            for idx, n in enumerate(nv):
                if n:
                    jdx = np.where(NC.values[idx, :] == np.sort(NC.values, axis = 1)[idx,-(i+2)])[0]
                    BC.iloc[idx, jdx] = True
        else:
            break
    
#    print('thresholded')    
#    print(BC)
    
    return BC

def match_trees(X, tree, y_true):
    '''
    Next, we check whether it is possible to match the trees. If either rowsums
    or colsums is bigger than 0, it is possible to match the trees (i.e. a 
    node is either merged or splitted in this scenario). 
    
    If the rowsums and colsums at a location are >1, this can happen, if this
    node is NOT a leaf node. 
    '''
    
    name_root1 = tree[0].name
    name_root2 = 'root2'

    binary = X > 0
#    print(binary.values)
    
    strict = X > 1
    
    # places where the root is involved, should be kept anyway 
    # they only give extra information and will never lead to a conflict
    # besides, they can never have a 2 
    tokeep = np.where(binary.index.values == name_root2)[0]
    strict.iloc[tokeep] = binary.iloc[tokeep]
    tokeep = np.where(binary.columns.values == name_root1)[0]
    strict.iloc[:,tokeep] = binary.iloc[:,tokeep]
    
    # scan binary matrix
    binary, y_true = scan_binary(binary, name_root1, name_root2, y_true, tree)

    strict.iloc[binary.values == False] = False
    
    # scan strict matrix if needed
    if (np.any(strict)):
#        print('\n\n\n Start strict')
        binary, y_true = scan_binary(strict, name_root1, name_root2, y_true, tree)
    
    return y_true

def scan_binary(binary, name_root1, name_root2, y_true, tree):
    
    colsum = np.sum(binary, axis = 0)
    rowsum = np.sum(binary, axis = 1)

    for idx, i in enumerate(colsum):
        for jdx, j in enumerate(rowsum):
                        
            if binary.iloc[jdx,idx]:
                binary, y_true = find_scenario(i,j,idx,jdx,rowsum, colsum,
                                               binary,name_root1,name_root2,
                                               y_true,tree)
    
    return binary, y_true
    


def find_scenario(i, j, idx, jdx, rowsum, colsum, binary, name_root1, name_root2, y_true, tree):
    '''
    Find the corresponding scenario between two matching labels
    
    Parameters
    ----------
    i: value of the colsum
    j: value of the rowsum
    idx: index of the column
    jdx: index of the  row
    rowsum: rowsum of the binary matrix
    colsum: colsum of the binary matrix
    binary: binary matrix
    name_root1: name of the root of the first tree
    name_root2: name of the root of the second tree
    y_true: labels of the second dataset that need to be changed in case of
            a perfect match
    tree: classification tree belonging to dataset 1, so this tree is updated
    
    Return
    ------
    binary: updated binary matrix
    y_true: updated labels    
    '''
        
    # Simple scenario
    if ((i == 1) & (j == 1)):
#        print('Simple')
        
        # Check if root1 is involved, if so -> new population
        if binary.columns.values[idx] == name_root1:
            newnode = Node(binary.index.values[jdx])
            tree[0].add_descendant(newnode)
            binary.iloc[jdx,idx] = False
        
        # Check if root2 is involved, if so -> do nothing
        elif binary.index.values[jdx] == name_root2:
            binary.iloc[jdx,idx] = False
        
        # Otherwise perfect match between two populations
        else:
            pred_value = binary.columns.values[idx]
            cluster_value = binary.index.values[jdx]
            print('Perfect match: ', cluster_value, 'is now:', pred_value)
            y_true[np.where(y_true == cluster_value)[0]] = pred_value
            binary.iloc[jdx,idx] = False
     
    
    # Multiple populations from dataset 2 involved
    elif ((i > 1) & (j == 1)):
#        print('Colsums')
        CP_D1 = binary.columns.values[idx] # cell population from dataset 1
        
        # Find cell populations from D2 involved and check if these cell
        # populations (except for the root) are involved in a complex scenario
        # if so -> skip
        CP_D2_idx = np.where(binary.iloc[:,idx] == True)[0]
        rowsum_CP_D2 = rowsum[CP_D2_idx]
        CP_D2_complex = rowsum_CP_D2.index.values[np.where(rowsum_CP_D2 > 1)[0]]
        CP_D2_complex = CP_D2_complex[np.where(CP_D2_complex != name_root2)[0]]

        if len(CP_D2_complex) > 0:
#            print('stop')
            return binary, y_true
        
        CP_D2 = binary.index.values[CP_D2_idx]
        split_node(tree, CP_D1, CP_D2, name_root2)
        binary.iloc[:,idx] = False

    
    # Multiple populations from dataset 1 involved
    elif (i == 1) & (j > 1):
#        print('rowsums')
        CP_D2 = binary.index.values[jdx]
        
        if CP_D2 == name_root2:
#            print('want to add root node 2, return')
            return binary, y_true
        
        # Find cell populations from D1 involved and check if these cell
        # populations (except for the root) are involved in a complex scenario
        # if so -> skip
        CP_D1_idx = np.where(binary.iloc[jdx] == True)[0]
        colsum_CP_D1 = colsum[CP_D1_idx]
        CP_D1_complex = colsum_CP_D1.index.values[np.where(colsum_CP_D1 > 1)[0]]
        CP_D1_complex = CP_D1_complex[np.where(CP_D1_complex != name_root1)[0]]

        if len(CP_D1_complex) > 0:
#            print('stop')
            return binary, y_true
        
        CP_D1 = binary.columns.values[CP_D1_idx]
#        print('To merge:', CP_D1)
#        print('To add:', CP_D2)
        y_true = merge_node(tree, CP_D1, CP_D2, y_true)
        binary.iloc[jdx,:] = False

    # Complex scenario, multiple CP from D1 and D2 involved
    else:
#        print('complex')
        CP_D2_idx = np.where(binary.iloc[:,idx] == True)[0]
        CP_D2 = binary.index.values[CP_D2_idx]
#        print('rownodes', CP_D2)
        
        CP_D1_idx = np.where(binary.iloc[jdx] == True)[0]
        CP_D1 = binary.columns.values[CP_D1_idx]
#        print('colnodes', CP_D1)
        
        # check if it is a very complex scenario (e.g. we have a square)
        CP_D2_rowsum = rowsum[CP_D2_idx]
        CP_D2_complex = CP_D2_rowsum.index.values[np.where(CP_D2_rowsum > 1)[0]]
        CP_D2_complex = CP_D2_complex[np.where(CP_D2_complex != name_root2)[0]]
        
        CP_D1_colsum = colsum[CP_D1_idx]
        CP_D1_complex = CP_D1_colsum.index.values[np.where(CP_D1_colsum > 1)[0]]
        CP_D1_complex = CP_D1_complex[np.where(CP_D1_complex != name_root1)[0]]
        
        if (len(CP_D2_complex) > 1) | (len(CP_D1_complex) > 1):
#            print('stop, very complex')
            return binary, y_true
        
        
        # If two CP from D1 involved and one of them is the root
        # -> neglect root1, just refine the boundary
        if (len(CP_D1) == 2) & np.isin(name_root1, CP_D1):
            CP_D1 = CP_D1[np.where(CP_D1 != name_root1)[0]]
            CP_D1 = CP_D1[0]
            split_node(tree, CP_D1, CP_D2, name_root2)
            binary.iloc[:,idx] = False
            binary.iloc[jdx] = False
            
            ## We should also set the places to false where row nodes 
            ## have match with root1
            root_idx = np.where(binary.columns.values == name_root1)[0]
            binary.iloc[CP_D2_idx,root_idx] = False
            
            return binary, y_true
        
        # Remove the nodes where root1 is involved
        if (len(CP_D1) > 2) & np.isin(name_root1, CP_D1):
            CP_D1 = CP_D1[np.where(CP_D1 != name_root1)]
        
        # Check if the difficult node is a leaf node
        CP_D1_difficult = binary.columns.values[idx]
#        print('difficultnode', CP_D1_difficult)
        
        if np.isin(CP_D1_difficult, tree[0].get_leaf_names()) == False:
#            print('Not a leaf')
            
            # Check is CP_D1_difficult is the common ancestor of the rest of CP_D1
            count = 0
            for dn in tree[0].walk():
                if dn.name == CP_D1_difficult:
                    for dn2 in dn.walk():
                        if np.isin(dn2.name, CP_D1):
                            count += 1
                    if count == len(CP_D1):
            
                        split_node(tree, CP_D1_difficult, CP_D2, name_root2)
                        
                        CP_D1 = CP_D1[np.where(CP_D1 != CP_D1_difficult)[0]]
                        toadd = binary.index.values[jdx]
                        y_true = merge_node(tree, CP_D1, toadd, y_true)
                        binary.iloc[:,idx] = False
                        binary.iloc[jdx] = False
                        return binary, y_true
#                    else:
#                        print('Impossible now')
                    break
            
        else:
#            print('It is a leaf')
            if (len(CP_D2) == 2) & np.isin(name_root2, CP_D2):
                CP_D2 = CP_D2[np.where(CP_D2 != name_root2)[0]]
                CP_D2 = CP_D2[0]
                y_true = merge_node(tree, CP_D1, CP_D2, y_true)
                binary.iloc[:,idx] = False
                binary.iloc[jdx] = False
                return binary, y_true
#            else:
#                print('Impossible now')

            
    return binary, y_true
        

def split_node(tree, CP_D1, CP_D2, name_root2):
    '''
    Function to split an existing node
    
    Parameters
    ----------
    tree: the tree that should be updated
    CP_D1: name (string) of the cell population of D1 that is involved
    CP_D2: list containing the name(s) (string) of the cell populations of D2
           that should be added as children to CP_D1
    name_root2: the name of root2, if this node is involved, it should be skipped
    '''
    
    ### Remove root2 from the toadd if needed
    not_root2_idx = np.where(CP_D2 != name_root2)[0]
    CP_D2 = CP_D2[not_root2_idx]
    
    for n in tree[0].walk(mode = 'postorder'):
        if(n.name == CP_D1):
            for name in CP_D2:
                newnode = Node(name)
                n.add_descendant(newnode)

            return  
        

def merge_node(tree, CP_D1, CP_D2, y_true):
    '''
    Function to merge existing (leaf) nodes
    
    Parameters
    ----------
    tree: the tree that should be updated
    CP_D1: list containing the name(s) (string) of the cell populations of D1
           that should be rewired as children to CP_D2
    CP_D2: name (string) of the cell population of D12 that should be added
           to the tree
    y_true: labels of dataset 2
    
    Return
    ------
    y_true: updated labels of dataset 2
    '''
#    print('Merge node')
    
    # In the complex scenario it can happen that CP_D2 is already in the tree
    # If so -> rewire CP_D2 to this node
    for n in tree[0].walk():
        if n.name == CP_D2:
            CP_D2_ancestor = n.ancestor
            CP_D2_node = n
            
            for i in CP_D2_ancestor.descendants:
                to_rewire = False
                if i.name != CP_D2:
                    for j in i.walk():
                        if np.isin(j.name, CP_D1):
                            to_rewire = True
                    if to_rewire:
                        add_merge_node(i, CP_D2_node, CP_D2_ancestor)

            return y_true
    
    
#    print('Check scenario 1')
    # MERGE SCENARIO 1
    # If root1 is in CP_D1, we should add an extra node between
    # the root and all other to CP_D1 nodes
    if np.isin(tree[0].name, CP_D1):
        common = tree[0].name
        tokeep = np.where(CP_D1 != tree[0].name)[0]
        CP_D1 = CP_D1[tokeep]
        
        CP_D2_node = Node(CP_D2)
        
        for i in tree[0].descendants:
            to_rewire = False           
            for j in i.walk():
                if np.isin(j.name, CP_D1):
                    to_rewire = True                      
            if to_rewire:
                add_merge_node(i, CP_D2_node, tree[0])
                        
        tree[0].add_descendant(CP_D2_node)
        return y_true
        
#    print('Check scenario 2')
    # MERGE SCENARIO 2
    # If there is one CP in CP_D1 that is the ancestor of all others in CP_D1,
    # we know we have a perfect match between this node and CP_D2 
    for n in tree[0].walk():
        if np.isin(n.name, CP_D1):
            count = 0
            count2 = 0
            common = n.name
#            print('Common', common)
            for c in n.walk():
                if np.isin(c.name, CP_D1):
                    count += 1
                count2 += 1
            if (count == len(CP_D1)) & (count == count2):
#                print('Scenario 2.1')
                ## We have a perfect match
                pred_value = common
                cluster_value = CP_D2
                print('Perfect match: ', cluster_value, 'is now:', pred_value)
                y_true[np.where(y_true == cluster_value)[0]] = pred_value
                return y_true
            else:
                ### We want to remove this node from the tomerge, because we 
                ### want to attach the nodes under it
                ### E.g. if CD4+ cells have a match with T-cells, CD4+ reg and
                ### CD4+ naive cells, we see that CD4+ and T-cells do not
                ### have a perfect match, so we want to add CD4+ between T-cells
                ### and CD4+ reg and CD4+ naive.
                ### Only if count == len(tomerge)
                if count == len(CP_D1):
#                    print('Scenario 2.2')
#                    print('Old to merge:', CP_D1)
                    xx = np.where(CP_D1 != common)[0]
                    CP_D1 = CP_D1[xx]
#                    print('New to merge:', CP_D1)
   
            break
    
    ## MERGE SCENARIO 3
    ## This is the classical merge
#    print('Check scenario 3')
    common = common_ancestor(CP_D1, tree)
    CP_D2_node = Node(CP_D2)
    
    for n in tree[0].walk():
        
        if n.name == common:
            
            # Check if the new node we want to add is redundant
            # If so -> perfect match
            counter2 = 0
            for i in n.descendants:
                if np.isin(i.name, CP_D1):
                    counter2 += 1
            
            if counter2 == len(n.descendants):
                y_true[np.where(y_true == CP_D2)[0]] = n.name
                return y_true
                            
            # Rewire the descendants of the common ancestor
            for i in n.descendants:
                to_rewire = False
                for j in i.walk():
                    if np.isin(j.name, CP_D1):
                        to_rewire = True
                if to_rewire:
                    add_merge_node(i, CP_D2_node, n)
            n.add_descendant(CP_D2_node)

    return y_true
        
    
def add_merge_node(n, new_ancestor, old_ancestor):  
    '''
    Function to rewire an existing node to a new ancestor
    
    Parameters
    ----------
    n: node that should be rewired
    new_ancestor: the new ancestor of n
    old_ancestor: the old ancestor of n
    '''

    
    #set new parent node
    n.ancestor = new_ancestor 
    new_ancestor.add_descendant(n) 
        
    #remove n from the previous parent
    olddescendants = old_ancestor.descendants
    old_ancestor.descendants = []
    for j in olddescendants:
        if j.name != n.name:
            old_ancestor.add_descendant(j)
        
    return

def common_ancestor(CP_D1, tree):
    '''
    Find the common ancestor of nodes in the tree
    
    Parameters
    ----------
    CP_D1: the names of the nodes
    tree: classification tree
    '''
    
    all_ancestors = []
    
    for n in tree[0].walk():
        if np.isin(n.name, CP_D1):
            ## Find the ancestor of this node and store them
            ancestors = []
            a = n.ancestor
            while a != None:
#                print(a)
                ancestors.append(a.name)
                a = a.ancestor
            all_ancestors.append(ancestors)
    
#    print(all_ancestors)
    common = all_ancestors[0]
    
    if len(all_ancestors) > 1: 
    
        for i in range(1,len(all_ancestors)):
            common, ind1, ind2 = np.intersect1d(common, all_ancestors[i], return_indices=True)
    
        common_ancestor = all_ancestors[i][np.min(ind2)]
    
    else:
        common_ancestor = common
    
    return common_ancestor