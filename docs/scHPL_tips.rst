A few tips for using scHPL
===================

This page will be updated soon with more tips. If you have questions in the mean time, just open a `GitHub issue <https://github.com/lcmmichielsen/scHPL/issues/new/choose>`_ or send an email to l.c.m.michielsen 'at' tudelft.nl 

Which classifier to use?
-----------------------

We advise to use:

- the linear SVM when your integrated data still has a lot of dimensions (e.g. when you have used Seurat to integrate the datasets)
- the kNN when your integrated data has less, 10-50, dimensions (e.g. when you have used scVI or Harmony to integrate the datasets)
- the one-class SVM when your main focus is to find unseen cell populations. A downside of the one-class SVM, however, is that the classification performance drops.

Preparing your AnnData object
---------------------------

The input for the learn function is an AnnData object where the labels and batch-indicators are a column in the metadata. If you integrated your data, it can be that it's a different slot in the object. At the moment, it is NOT possible to indicate which slot to use for scHPL. Therefore, we advise to make a new AnnData object and copy the integrated data to the '.X'.

treeArches
---------

One way to integrate your data is with `treeArches <https://doi.org/10.1101/2022.07.07.499109>`_. treeArches is a wrapper around scHPL and scArches to easily create and update reference atlases and the corresponding hierarchy. There are two tutorials explaining how to use treeArches. 







