# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="schedulefree",
    version="1.3",
    author="Aaron Defazio",
    author_email="adefazio@meta.com",
    description="Schedule Free Learning in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/schedule_free",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
)