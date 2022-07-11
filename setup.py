# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:24:15 2021

@author: lcmmichielsen
"""

from pathlib import Path

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scHPL", 
    version="1.0.1",
    author="Lieke Michielsen",
    author_email="l.c.m.michielsen@tudelft.nl",
    description="Hierarchical progressive learning pipeline for single-cell RNA-sequencing datasets",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lcmmichielsen/scHPL",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "scipy>=1.5.2",
        "scikit-learn>=0.23.2",
        "pandas>=1.1.2",
        "newick~=1.0.0",
        "anndata>=0.7.4",
        "matplotlib>=3.3.1",
        "seaborn>=0.11.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    doc=[
          'sphinx',
          'sphinx_rtd_theme',
          'sphinx_autodoc_typehints',
          'typing_extensions; python_version < "3.8"',
      ],
)


