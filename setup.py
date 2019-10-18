# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():
    return read("VERSION").strip()


# Declare minimal set for installation
required_packages = [
    "torch>=1.2.0",
    "torchvision>=0.3",
    "future>=0.15.0"
]

setup(
    name='adatune',
    packages=['adatune'],
    package_dir={'adatune': 'adatune'},
    version=read_version(),
    description="Open source library to perform gradient based HPO for Deep Learning models",
    url="https://github.com/awslabs/adatune",
    author="Amazon Web Services",
    license="Apache License 2.0",
    keywords="ML Amazon AWS AI PyTorch AutoML HPO",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
    install_requires=required_packages
)
