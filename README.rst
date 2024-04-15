|PyPI| |PyPIDownloads| |Docs|

scHPL: Hierarchical progressive learning of cell identities in single-cell data


.. raw:: html

 <img src="https://github.com/lcmmichielsen/scHPL/blob/master/docs/scHPL.png" width="820px" align="center">

We present a hierarchical progressive learning method which automatically finds relationships between cell populations across multiple datasets and uses this to construct a hierarchical classification tree. For each node in the tree either a linear SVM, kNN, or one-class SVM, which enables the detection of unknown populations, is trained. The trained classification tree can be used to predict the labels of a new unlabeled dataset. 

NOTE: scHPL is not a batch correction tool, we advise to align the datasets before matching the cell populations. We advise doing this with scVI or scArches (see section treeArches below).

Installation
-------------

scHPL requires Python 3.6 or higher. The easiest way to install scHPL is through the following command::

    pip install scHPL

General usage
---------------

The ```tutorial.ipynb``` notebook explains the basics of scHPL. The `vignette folder </vignettes>`_ contains notebooks to reproduce the inter-dataset experiments. See the `documentation <https://schpl.readthedocs.io/en/latest/>`_ for more information.

treeArches
-----------

treeArches is a framework around scHPL and `scArches <https://github.com/theislab/scarches>`_ to automatically build and update reference atlases and the classification tree. Examples can be found in the `treeArches reprodicibility Github <https://github.com/lcmmichielsen/treeArches-reproducibility>`_ and in this `notebook <https://github.com/theislab/scarches/blob/master/notebooks/treeArches_pbmc.ipynb>`_.

Datasets
---------

All datasets used are publicly available data and can be downloaded from Zenodo. The simulated data and aligned datasets used during the interdataset experiments can be downloaded from the `scHPL Zenodo <https://doi.org/10.5281/zenodo.4557712>`_. The filtered PBMC-FACS and AMB2018 dataset can be downloaded from the `scRNA-seq benchmark Zenodo <https://doi.org/10.5281/zenodo.3357167>`_

For citation and further information please refer to: `"Hierarchical progressive learning of cell identities in single-cell data" <https://www.nature.com/articles/s41467-021-23196-8>`_
 


.. |PyPI| image:: https://img.shields.io/pypi/v/scHPL.svg
   :target: https://pypi.org/project/scHPL

.. |PyPIDownloads| image:: https://static.pepy.tech/badge/scHPL
   :target: https://pepy.tech/project/scHPL

.. |Docs| image:: https://readthedocs.org/projects/schpl/badge/?version=latest
   :target: https://schpl.readthedocs.io
