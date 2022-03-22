Introduction
============

We present ``scHPL``, a hierarchial progressive learning method which automatically finds relationships between cell populations across multiple datasets and uses this to construct a hierarchical classification tree. For each node in the tree either a linear SVM or one-class SVM, which enables the detection of unknown populations, is trained. The trained classification tree can be used to predict the labels of a new unlabeled dataset.

NOTE: scHPL is not a batch correction tool, we advise to align the datasets before matching the cell populations using scHPL.

