# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from setuptools import setup, find_packages

setup(
    name="llmsdk",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "typing",
        "pydantic",
        "pandas",
        "openai>=0.27.4",
        'tiktoken',
        "tenacity",
    ],
)
