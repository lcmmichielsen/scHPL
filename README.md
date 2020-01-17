# Hierarchical progressive learning

### Pipeline
```train_hierarchical_classifier.py``` contains a wrapper function ```train_hierarchical_classifier``` which uses multiple labeled datasets to construct a classification tree. This function uses ```build_classifier``` to train a hierarchical tree, ```predict_labels``` to predict the labels of a dataset, and ```update_tree``` to match the labels of two datasets and update the tree.

All functions rely on ```newick.py```, which is used to read, write, and store the classification trees. The function is adapted from https://github.com/glottobank/python-newick/blob/master/src/newick.py such that it is possible to save a trained classifier and the discriminating principal components per node. 

### Evaluation
The ```hierarchical_F1.py``` can be used to calculate the hierarchical F1-score and evaluate the classification.
