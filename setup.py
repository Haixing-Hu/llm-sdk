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
        "numpy",
        "frozendict",
        "requests",
        "pydantic",
        "pandas",
        "openai>=0.27.4",
        "tenacity",
        "parameterized",
    ],
)
