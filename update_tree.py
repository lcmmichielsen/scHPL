# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:58:14 2019

@author: Lieke
"""

import numpy as np
import pandas as pd
from newick import *
from sklearn.metrics import jaccard_score

def update_tree(y_cluster1, y_pred1, y_cluster2, y_pred2, threshold, tree):
    '''
    Match the labels of two datasets 
    
    Parameters
    ----------
    y_cluster1: cluster labels of dataset 1 
    y_pred1: the predictions made on dataset1, so made by tree2
    y_cluster2: cluster labels of dataset 2 
    y_pred2: the predictions made on dataset2, so made by tree1
    threshold: threshold used to create the binary confusion matrix
    tree: tree belonging to dataset 1
    
    Return
    ------
    tree: the update tree
    y_cluster2: the updated labels of y_cluster 2 so that they correspond
    to the labels of y_cluster1 in case of a perfect match
    '''
        
    # First make the two confusion matrices and add them
    conf1 = confusion_binary(y_cluster1, y_pred1, threshold)
    conf1 = conf1.reindex(index = tree[0].get_leaf_names(), fill_value = False)
    
    conf2 = confusion_binary(y_cluster2, y_pred2, threshold)
    conf_combined = conf2.add(conf1.T, fill_value = 0)
    
    
    # Use the added matrix to find matches
    y_cluster2 = match_trees(conf_combined, tree, y_cluster2)
    
        
    return tree, y_cluster2
    
def confusion_binary(y_cluster, y_pred, threshold):
    '''
    Construct a binary confusion matrix using the Jaccard score
    '''
    
    num_cluster = len(np.unique(y_cluster))
    num_pred = len(np.unique(y_pred))

    conf = np.zeros([num_cluster,num_pred])

    for i, group_true in enumerate(np.unique(y_cluster)):
        a = y_cluster == group_true
        a = np.squeeze(a)
        for j, group_pred in enumerate(np.unique(y_pred)):
            b = y_pred == group_pred
            b = np.squeeze(b)
            conf[i,j] = sum(a & b)/ sum(a)

    conf = pd.DataFrame(conf, columns = np.unique(y_pred), index = np.unique(y_cluster))
    print(conf.values)
    
    th = conf > 1
    
    ## First we find the max value and put a True there
    max_values = np.argmax(conf.values, axis = 1)
    for idx, mv in enumerate(max_values):
        th.iloc[idx,mv] = True
        
    nv = np.ones([np.shape(conf)[0],], dtype = bool)
    print(np.shape(nv))
    ## Next, we look if the difference between the second highest and highest
    ## is smaller than the threshold. (and third highest than the second etc.)
    for i in range((np.shape(conf)[1] - 2)):
        sort1 = np.sort(conf.values, axis = 1)[:,-(i+1)]
        sort2 = np.sort(conf.values, axis = 1)[:,-(i+2)]
        new = sort1 - sort2
        nv = nv & (new < threshold) & (sort2 > 0.1)
        if np.sum(nv) > 0:
            for idx, n in enumerate(nv):
                if n:
                    jdx = np.where(conf.values[idx, :] == np.sort(conf.values, axis = 1)[idx,-(i+2)])[0]
                    th.iloc[idx, jdx] = True
        else:
            break
    
    print('thresholded')    
    print(th)
    
    return th

def match_trees(conf, tree, y_cluster):
    '''
    Next, we check whether it is possible to match the trees. If either rowsums
    or colsums is bigger than 0, it is possible to match the trees (i.e. a 
    node is either merged or splitted in this scenario). 
    
    If the rowsums and colsums at a location are >1, this can happen, if this
    node is NOT a leaf node. 
    '''
    
    name_root1 = tree[0].name
    name_root2 = 'root2'

    binary = conf > 0
    print(binary.values)
    
    strict = conf > 1
    
    # places where the root is involved, should be kept anyway 
    # they only give extra information and will never lead to a conflict
    # besides, they can never have a 2 
    tokeep = np.where(binary.index.values == name_root2)[0]
    strict.iloc[tokeep] = binary.iloc[tokeep]
    tokeep = np.where(binary.columns.values == name_root1)[0]
    strict.iloc[:,tokeep] = binary.iloc[:,tokeep]
    
    binary, y_cluster = scan_binary(binary, name_root1, name_root2, y_cluster, tree)

    strict.iloc[binary.values == False] = False
    
    if (np.any(strict)):
        print('\n\n\n Start strict')
        binary, y_cluster = scan_binary(strict, name_root1, name_root2, y_cluster, tree)
    
    return y_cluster

def scan_binary(binary, name_root1, name_root2, y_cluster, tree):
    
    colsum = np.sum(binary, axis = 0)
    rowsum = np.sum(binary, axis = 1)

    for idx, i in enumerate(colsum):
        for jdx, j in enumerate(rowsum):
                        
            if binary.iloc[jdx,idx] == False:
                continue
        
            
            #### Look at the different scenarios
            binary, y_cluster = find_scenario(i,j,idx,jdx,rowsum, colsum,binary,name_root1,name_root2,y_cluster,tree)
    
    return binary, y_cluster
    


def find_scenario(i,j,idx,jdx,rowsum,colsum,binary,name_root1,name_root2,y_cluster,tree):
        
    #### Simple scenario
    if ((i == 1) & (j == 1)):
        print('Simple')
        if binary.columns.values[idx] == name_root1:
            newnode = Node(binary.index.values[jdx])
            tree[0].add_descendant(newnode)
            binary.iloc[jdx,idx] = False
        elif binary.index.values[jdx] == name_root2:
            binary.iloc[jdx,idx] = False
        else:
            # perfect match
            pred_value = binary.columns.values[idx]
            cluster_value = binary.index.values[jdx]
            y_cluster[np.where(y_cluster == cluster_value)[0]] = pred_value
            binary.iloc[jdx,idx] = False
     
    #### Only colsums, so we want to split a node in the tree
    elif ((i > 1) & (j == 1)):
        print('Colsums')
        tosplit = binary.columns.values[idx]
        row = np.where(binary.iloc[:,idx] == True)[0]
        rows = rowsum[row]
        rownames = rows.index.values[np.where(rows > 1)[0]]
        rownames = rownames[np.where(rownames != name_root2)[0]]
        
        # check if there is a complex scenario somewhere else in this column
        # we don't want to solve this now, but do it later
        if len(rownames) > 0:
            print('stop')
            return binary, y_cluster
        
        toadd = binary.index.values[row]
        split_node(tree, tosplit, toadd, name_root2)
        binary.iloc[:,idx] = False
    
    #### Only rowsums, so we want to merge nodes in the tree
    elif (i == 1) & (j > 1):
        print('rowsums')
        toadd = binary.index.values[jdx]
        
        if toadd == name_root2:
            print('want to add root node 2, return')
            return binary, y_cluster
        
        col = np.where(binary.iloc[jdx] == True)[0]
        cols = colsum[col]
        colnames = cols.index.values[np.where(cols > 1)[0]]
        colnames = colnames[np.where(colnames != name_root1)[0]]

        
        # check if there is a complex scenario somewhere else in this row
        # we don't want to solve this now, but do it later
        if len(colnames) > 0:
            print('stop')
            return binary, y_cluster
        
        tomerge = binary.columns.values[col]
        print('To merge:', tomerge)
        print('To add:', toadd)
        y_cluster = merge_node(tree, tomerge, toadd, y_cluster)
        binary.iloc[jdx] = False

    #### Complex scenario, the row and colsums are biggen than 1
    else:
        print('complex')
        row = np.where(binary.iloc[:,idx] == True)[0]
        rownodes = binary.index.values[row]
        print('rownodes', rownodes)
        
        col = np.where(binary.iloc[jdx] == True)[0]
        colnodes = binary.columns.values[col]
        print('colnodes', colnodes)
        
        # We neglect root1 and just refine the boundary
        # Becomes simple splitting scenario
        if (len(colnodes) == 2) & np.isin(name_root1, colnodes):
            colnodes = colnodes[np.where(colnodes != name_root1)[0]]
            tosplit = colnodes[0]
            toadd = rownodes
            split_node(tree, tosplit, toadd, name_root2)
            binary.iloc[:,idx] = False
            binary.iloc[jdx] = False
            
            ## We should also set the places to false where row nodes 
            ## have match with root1
            root_idx = np.where(binary.columns.values == name_root1)[0]
            binary.iloc[row,root_idx] = False
            
            return binary, y_cluster
        
        # Just ignore this, the boundary should be refined, but that 
        # happens automatically
        if (len(colnodes) > 2) & np.isin(name_root1, colnodes):
            print(colnodes)
            colnodes = colnodes[np.where(colnodes != name_root1)]
            print(colnodes)
        
        # Check if the difficult node is a leaf node
        difficultnode = binary.columns.values[idx]
        print('difficultnode', difficultnode)
        if np.isin(difficultnode,tree[0].get_leaf_names()) == False:
            print('Not a leaf')
            
            # Check is the difficult node is the common ancestor of the others
            count = 0
            for dn in tree[0].walk():
                if dn.name == difficultnode:
                    for dn2 in dn.walk():
                        if np.isin(dn2.name, colnodes):
                            count += 1
                    if count == len(colnodes):
            
                        # Split difficult node in the rownodes
                        tosplit = difficultnode
                        # we don't want to add the tomerge node here, we will do that later
                        toadd = rownodes
                        split_node(tree, tosplit, toadd, name_root2)
                        
                        # Find nodes that should be rewired to this newly added node
                        tomerge = colnodes[np.where(colnodes != tosplit)[0]]
                        toadd = binary.index.values[jdx]
                        y_cluster = merge_node(tree, tomerge, toadd, y_cluster)
                        binary.iloc[:,idx] = False
                        binary.iloc[jdx] = False
                        return binary, y_cluster
                    else:
                        print('Impossible now')
                    break
            
        else:
            print('It is a leaf')
            if (len(rownodes) == 2) & np.isin(name_root2, rownodes):
                tomerge = colnodes
                rownodes = rownodes[np.where(rownodes != name_root2)[0]]
                toadd = rownodes[0]
                y_cluster = merge_node(tree, tomerge, toadd, y_cluster)
                binary.iloc[:,idx] = False
                binary.iloc[jdx] = False
                return binary, y_cluster
            else:
                print('Impossible now')

            
    return binary, y_cluster
        

def split_node(tree, tosplit, toadd, name_root2):
    '''
    Function to split an existing node
    
    tree: the tree containing the node that should be splitted
    tosplit: name (string) of the node that should be splitted
    toadd: list containing the name(s) (string) of the node(s) 
    that should be added
    '''
    
    ### Remove root2 from the toadd if needed
    not_root2_idx = np.where(toadd != name_root2)[0]
    toadd = toadd[not_root2_idx]
    
    for n in tree[0].walk(mode = 'postorder'):
        if(n.name == tosplit):
            add_leaf_nodes(n, toadd)
            return  
        
                
def add_leaf_nodes(node_tosplit, toadd):
    '''
    Function to attach the leaf nodes (toadd) to the correct ancestor
    (node_tosplit).
    '''
    
    for name in toadd:
        newnode = Node(name)
        node_tosplit.add_descendant(newnode)
            
def merge_node(tree, tomerge, toadd, y_cluster):
    '''
    Function to merge existing (leaf) nodes
    
    tree: the tree containing the node that should be splitted
    tomerge: list of names (string) of the nodes that should be merged
    toadd: name (string) of the node that should be added
    '''
    print('Merge node')
    
    # FIRST LOOK IF TOADD IS ALREADY IN THE TREE 
    # (this can happen in the complex scenario)
    for n in tree[0].walk():
        if n.name == toadd:
            # in this case we should just rewire the tomerge nodes to this node
            ancestor = n.ancestor
            newnode = n
            
            for i in ancestor.descendants:
                to_rewire = False
                if i.name == toadd:
                    continue
                else:
                    for j in i.walk():
                        if np.isin(j.name, tomerge):
                            to_rewire = True
                    if to_rewire:
                        add_merge_node(i, newnode, ancestor)

            return y_cluster
    
    print('Check scenario 1')
    # MERGE SCENARIO 1
    # If the root is in tomerge, we should add an extra node between
    # the root and all other to merge nodes
    if np.isin(tree[0].name, tomerge):
        common = tree[0].name
        tokeep = np.where(tomerge != tree[0].name)[0]
        tomerge = tomerge[tokeep]
        
        newnode = Node(toadd)
        
        for i in tree[0].descendants:
            # Check if i is a descendant that should be reconnected 
            # to the new node (if the node itself or one of its descendants
            # is in the tomerge list)
            to_rewire = False
                        
            for j in i.walk():
                if np.isin(j.name, tomerge):
                    to_rewire = True
                                        
            if to_rewire:
                add_merge_node(i, newnode, tree[0])
                        
        tree[0].add_descendant(newnode)
        return y_cluster

    ## MERGE SCENARIO 2
    ## If there is one node in tomerge that is the ancestor of all other nodes
    ## in tomerge, we know we have a perfect match between this node and the 
    ## toadd node, so we should only change the labels
    print('Check scenario 2')
    for n in tree[0].walk():
        if np.isin(n.name, tomerge):
            count = 0
            count2 = 0
            common = n.name
            print('Common',common)
            for c in n.walk():
                if np.isin(c.name, tomerge):
                    count += 1
                count2 += 1
            if (count == len(tomerge)) & (count == count2):
                print('Scenario 2.1')
                ## We have a perfect match
                pred_value = common
                cluster_value = toadd
                y_cluster[np.where(y_cluster == cluster_value)[0]] = pred_value
                return y_cluster
            else:
                ### We want to remove this node from the tomerge, because we 
                ### want to attach the nodes under it
                ### E.g. if CD4+ cells have a match with T-cells, CD4+ reg and
                ### CD4+ naive cells, we see that CD4+ and T-cells do not
                ### have a perfect match, so we want to add CD4+ between T-cells
                ### and CD4+ reg and CD4+ naive.
                ### Only if count == len(tomerge)
                if count == len(tomerge):
                    print('Scenario 2.2')
                    print('Old to merge:', tomerge)
                    xx = np.where(tomerge != common)[0]
                    tomerge = tomerge[xx]
                    print('New to merge:', tomerge)
                
                
            break
    
    ## MERGE SCENARIO 3
    ## This is the classical merge
    print('Check scenario 3')
    common = common_ancestor(tomerge, tree)
    newnode = Node(toadd)
    
    for n in tree[0].walk():
        
        # First find the common ancestor in the tree
        if n.name == common:
            
            # Check if the new node we want to add is redundant
            counter2 = 0
            for i in n.descendants:
                if np.isin(i.name, tomerge):
                    counter2 += 1
                        
            # If the node already exists
            if counter2 == len(n.descendants):
                # Change cluster labels to this nodes name
                y_cluster[np.where(y_cluster == toadd)[0]] = n.name
                return y_cluster
                            
            # Rewire the descendants of the common ancestor
            for i in n.descendants:
                # Check if i is a descendant that should be reconnected 
                # to the new node (if the node itself or one of its descendants
                # is in the tomerge list)
                to_rewire = False
                    
                for j in i.walk():
                    if np.isin(j.name, tomerge):
                        to_rewire = True
                                    
                if to_rewire:
                    add_merge_node(i, newnode, n)
                    
            n.add_descendant(newnode)

    return y_cluster
        
    
def add_merge_node(n, newnode, ancestor):  
    
    #set new parent node
    n.ancestor = newnode 
    newnode.add_descendant(n) 
        
    #remove n from the previous parent
    olddescendants = ancestor.descendants
    ancestor.descendants = []
    for j in olddescendants:
        if j.name != n.name:
            ancestor.add_descendant(j)
        
    return

def common_ancestor(tomerge, tree):
    
    all_ancestors = []
    
    for n in tree[0].walk():
        if np.isin(n.name, tomerge):
            ## Find the ancestor of this node and store them
            ancestors = []
            a = n.ancestor
            while a != None:
                print(a)
                ancestors.append(a.name)
                a = a.ancestor
            all_ancestors.append(ancestors)
    
    print(all_ancestors)
    common = all_ancestors[0]
    
    if len(all_ancestors) > 1: 
    
        for i in range(1,len(all_ancestors)):
            common, ind1, ind2 = np.intersect1d(common, all_ancestors[i], return_indices=True)
    
        common_ancestor = all_ancestors[i][np.min(ind2)]
    
    else:
        common_ancestor = common
    
    return common_ancestor