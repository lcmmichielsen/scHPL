# Inter-dataset experiment

During this vignette we will repeat the PBMC inter-dataset experiment. We use three datasets to construct a classification tree and use this tree to predict the labels of a fourth dataset. The aligned datasets and labels can be downloaded from http://doi.org/10.5281/zenodo.3736493

### Read the data

We start with reading the different datasets and corresponding labels and adding them to a list.

In the datasets, the rows represent different cells and columns represent the genes.


```python
import pandas as pd
import numpy as np

path = 'path_to_files/'

data0 = 'Data_EQTL.csv'
labels0 = 'Labels_EQTL.csv'

data1 = 'Data_10Xv2.csv'
labels1 = 'Labels_10Xv2.csv'

data2 = 'Data_FACS.csv'
labels2 = 'Labels_FACS.csv'

data = []
labels = []

data.append((pd.read_csv(path + data0, index_col=0, sep=',')))
labels.append(pd.read_csv(path + labels0, header=0, index_col=None, sep=','))

data.append((pd.read_csv(path + data1, index_col=0, sep=',')))
labels.append(pd.read_csv(path + labels1, header=0, index_col=None, sep=','))

data.append((pd.read_csv(path + data2, index_col=0, sep=',')))
labels.append(pd.read_csv(path + labels2, header=0, index_col=None, sep=','))
```

### Construct and train the classification tree

Next, we use hierarchical progressive learning to construct and train a classification tree. After each iteration, an updated tree will be printed. If two labels have a perfect match, one of the labels will not be visible in the tree. Therefore, we will also indicate these perfect matches using a print statement

During this experiment, we used the linear SVM, didn't apply dimensionality reduction and used the default threshold of 0.25. In you want to use a one-class SVM instead of a linear, the following can be used: classifier = 'svm_occ'


```python
from train_hierarchical_classifier import *

tree = train_hierarchical_classifier(data, labels, classifier = 'svm', 
                                     dimred = False, threshold = 0.25)
```

    Iteration  1 
    
    Perfect match:  B cell - B-10Xv2 is now: B cell - eQTL
    Perfect match:  CD4+ T cell - B-10Xv2 is now: CD4+ T cell - eQTL
    Perfect match:  CD8+ T cell - B-10Xv2 is now: CD8+ T cell - eQTL
    
    Updated tree:
    Root: Node("root")
    B cell - eQTL
    CD4+ T cell - eQTL
    CD8+ T cell - eQTL
    pDC - eQTL
    Monocyte - B-10Xv2
       CD14+ Monocyte - eQTL
       CD16+ Monocyte - eQTL
       mDC - eQTL
    NK cell - B-10Xv2
       CD56+ bright NK cell - eQTL
       CD56+ dim NK cell - eQTL
    Megakaryocyte - B-10Xv2
       Megakaryocyte - eQTL    
    
    Iteration  2 
    
    Perfect match:  B cell - FACS is now: B cell - eQTL
    
    Updated tree:
    Root: Node("root")
    B cell - eQTL
    CD4+ T cell - eQTL
       CD4+/CD25 T Reg - FACS
       CD4+/CD45RA+/CD25- Naive T - FACS
       CD4+/CD45RO+ Memory - FACS
       CD8+/CD45RA+ Naive Cytotoxic - FACS
    CD8+ T cell - eQTL
       NK cell - FACS
    pDC - eQTL
    Monocyte - B-10Xv2
       CD14+ Monocyte - eQTL
       CD16+ Monocyte - eQTL
       mDC - eQTL
    NK cell - B-10Xv2
       CD56+ bright NK cell - eQTL
       CD56+ dim NK cell - eQTL
    Megakaryocyte - B-10Xv2
       Megakaryocyte - eQTL
    CD34+ cell - FACS
    
    


### Predict the labels of the testset

In this last step, we use the trained classification tree to predict the labels of a test set.


```python
from predict_labels import *

data3 = 'Data_10Xv3.csv'
data_test = pd.read_csv(path + data3, index_col=0, sep=',')

pred_test = predict_labels(data_test, tree)
```

We can compare these true and predicted labels by constructing a confusion matrix


```python
def conf(l_true, l_pred):
    
    num_cluster = len(np.unique(l_true))
    num_pred = len(np.unique(l_pred))

    conf = np.zeros([num_cluster,num_pred])

    for i, group_true in enumerate(np.unique(l_true)):
        a = l_true == group_true
        a = np.squeeze(a)
        for j, group_pred in enumerate(np.unique(l_pred)):
            b = l_pred == group_pred
            b = np.squeeze(b)
            conf[i,j] = sum(a & b) / sum(a)

    conf = pd.DataFrame(conf, columns = np.unique(l_pred), index = np.unique(l_true))
    
    return conf


labels3 = 'Labels_10Xv3.csv'
y_true = pd.read_csv(path + labels3, header=0, index_col=None, sep=',')
y_pred = pd.DataFrame(data = pred_test)

conf_lin = conf(y_true, y_pred)

```

This confusion matrix can be visualized using a heatmap.

In this heatmap, we change the order of the columns, such that it will be easier to compare the true and predicted cell populations


```python
import seaborn as sns
import matplotlib.pyplot as plt

order = np.array([0,14,1,2,4,5,6,7,10,11,16,17,12,13,15,8,9,3,18])

plt.figure(figsize=(12,4.5))
sns.heatmap(round(conf_lin,2).iloc[:,order], vmin = 0, vmax = 1, annot=True)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f47ec094668>




![png](output_10_1.png)


### Sessioninfo


```python
pip_modules = !pip freeze

includes = ['pandas', 'numpy', 'scipy', 'scikit-learn', 'seaborn', 'matplotlib']

#print the names and versions of the imported modules
for module in pip_modules:
    name, version = module.split('==')
    if name in includes: 
        print(name + '\t' + version)
```

    matplotlib	3.2.1
    numpy	1.18.2
    pandas	1.0.3
    scikit-learn	0.22.2.post1
    scipy	1.4.1
    seaborn	0.10.0



```python

```
