# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from setuptools import setup, find_packages

setup(
    name="llmsdk",
    version="0.3.3",
    packages=find_packages(),
    install_requires=[
        "numpy~=1.25.0",
        "frozendict",
        "requests",
        "tenacity",
        "cachetools",
        "tqdm",
    ],
)
