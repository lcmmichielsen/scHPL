|PyPI| |Docs|

scHPL
=========================================================================
.. raw:: html

 <img src="https://github.com/lcmmichielsen/scHPL/blob/master/docs/scHPL.png" width="820px" align="center">
 
scHPL (single-cell Hierarchical Progressive Learning) is method that automatically finds relationships between cell populations across multiple datasets and uses this to construct a hierarchical classification tree. For each node in the tree either a linear SVM, kNN, or one-class SVM is trained. The trained classification tree can be used to predict the labels of a new unlabeled dataset.

.. note::
   scHPL is not a batch correction tool, we advise to align the datasets before matching the cell populations. We advise doing this with scVI or scArches. See the treeArches tutorials or the 'tips' tab for more information.


.. |PyPI| image:: https://img.shields.io/pypi/v/scHPL.svg
   :target: https://pypi.org/project/scHPL

.. |Docs| image:: https://readthedocs.org/projects/schpl/badge/?version=latest
   :target: https://schpl.readthedocs.io


.. toctree::
   :maxdepth: 1
   :caption: Main
   :hidden:

   about
   installation
   api/index.rst
   scHPL_tips

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   tutorial
   treeArches_pbmc
   treeArches_identifying_new_ct
   AMB-inter-dataset
   brain-inter-dataset
   pbmc-inter-dataset
