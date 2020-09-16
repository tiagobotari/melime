#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name="melime",
        version="0.1",
        author="Tiago Botari, Frederik Hvilshøj",
        url="https://github.com/tiagobotari/melime/",
        author_email="",
        license="MIT License",
        packages=find_packages(),
        zip_safe=False,
        install_requires=[
            "scikit-learn>=0.21",
            "torch>=1.3",
            "torchvision>=0.4",
            "numpy>=1.17",
            "matplotlib",
            "scikit-image>=0.17.2",
            "gensim",
            ]
        , python_requires=">=3.6"
    )
