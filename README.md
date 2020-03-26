# Hierarchical progressive learning of cell identities in single-cell data

We present a hierarchical progressive learning method which automatically finds relationships between cell populations across multiple datasets and uses this to construct a hierarchical classification tree. For each node in the tree either a linear SVM or one-class SVM, which enables the detection of unknown populations, is trained. The trained classification tree can be used to predict the labels of a new unlabeled dataset. 

### General usage
If you have multiple labeled dataset, progressive learning can be used to match the labels of the datasets and train a classification tree on all datasets. In case of one labeled dataset, the classification tree can be trained on one dataset without progressive learning. In both cases, the function returns a trained classification tree. 

#### Training a classifier without progressive learning
First, a tree needs to be constructed for the dataset. This can be done using the ```loads``` function from ```newick.py```. An example to construct a classification tree for a PBMC data is shown below:

```Python
from newick import *
tree = loads('(CD34+, CD14+ Monocyte, ((CD19+ B, ((CD8+/CD45RA+ Naive Cytotoxic)CD8+ Cytotoxic T, (CD4+/CD25 T Reg, CD4+/CD45RA+/CD25- Naive T, CD4+/CD45RO+ Memory)CD4+ T Helper2)T cells)small lymphocytes, CD56+ NK)lymphocytes)root')
```
When constructing the tree, it is important that the names of the nodes are written exactly the same as the labels in the dataset.

Next, the ```train_tree``` function from ```build_classifier.py``` can be used to train this tree:

```Python
from build_classifier import *
tree = train_tree(data, labels, tree, classifier = 'svm_occ', dimred = True)
```
This function assumes that data is matrix (cells x genes) and the labels are a 1D vector. If you want to train the tree with a linear SVM for each node instead of a one-class SVM, ```classifier = 'svm'``` can be used. The parameter ```dimred``` indicates whether dimensionality reduction is applied.

#### Training a classifier with progressive learning
If you have multiple labeled datasets, the ```train_hierarchical_classifier``` function from ```train_hierarchical_classifier.py``` can be used to construct and train a classification tree:

```Python
from train_hierachical_classifier import *
tree = train_hierarchical_classifier(data, labels, classifier = 'svm_occ', dimred = True, threshold = 0.25)
```

This function assumes that data is a list containing all *n* datasets (n x cells x genes) and that labels contains the labels of all *n* datasets (n x cells). If you want to train the tree with a linear SVM for each node instead of a one-class SVM, ```classifier = 'svm'``` can be used. The parameter ```dimred``` indicates whether dimensionality reduction is applied. ```threshold``` indicates which threshold to use when matching the labels of two datasets.


#### Prediction
The trained classification tree can be used to predict the labels of a new dataset using the ```predict_labels``` function from the ```predict_labels.py``` script:

```Python
from predict_labels import *
y_pred = predict_labels(test_data, tree)
```

This function uses the testdata (cells x genes) and trained classification tree and returns a vector with the predicted labels. 

### Evaluation
The ```hierarchical_F1``` function form ```hierarchical_F1.py``` can be used to calculate the hierarchical F1-score and evaluate the classification. 

```Python
from hierarchical_F1 import *
HF1 = hierarchical_F1(y_true, y_pred, tree)
```

All functions rely on ```newick.py```, which is used to read, write, and store the classification trees. The function is adapted from https://github.com/glottobank/python-newick/blob/master/src/newick.py such that it is possible to save a trained classifier and the discriminating principal components per node. 

