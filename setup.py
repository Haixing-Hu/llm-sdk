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
    version="0.3.5",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.25.0",
        "frozendict",
        "requests",
        "tenacity",
        "cachetools",
        "tqdm",
    ],
    author='Haixing Hu',
    author_email='starfish.hu@gmail.com',
    description='A toolkit for developing applications with LLM (Large Language Model).',
    url='https://github.com/Haixing-Hu/llm-sdk',
)
