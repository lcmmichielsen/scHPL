# Hierarchical progressive learning of cell identities in single-cell data

We present a hierarchical progressive learning method which automatically finds relationships between cell populations across multiple datasets and uses this to construct a hierarchical classification tree. For each node in the tree either a linear SVM or one-class SVM, which enables the detection of unknown populations, is trained. The trained classification tree can be used to predict the labels of a new unlabeled dataset. 

### Installation

scHPL requires Python 3.6 or 3.7. The easiest way to install scHPL is through the following command::

    pip install scHPL

### General usage

The ```tutorial.ipynb``` notebook explains the basics of scHPL. The [vignette folder](vignettes) contains notebooks to reproduce the experiments.  

### Datasets

All datasets used are publicly available data and can be downloaded from Zenodo. The simulated data and aligned datasets used during the interdataset experiments can be downloaded from the scHPL Zenodo (https://doi.org/10.5281/zenodo.4557712). The filtered PBMC-FACS and AMB2018 dataset can be downloaded from the scRNA-seq benchmark Zenodo (https://doi.org/10.5281/zenodo.3357167)

For citation and further information please refer to:
"Hierarchical progressive learning of cell identities in single-cell data" ([biorxiv](https://www.biorxiv.org/content/10.1101/2020.03.27.010124v2))
