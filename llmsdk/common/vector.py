# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import List, Union, TypeAlias
import numpy as np


Vector: TypeAlias = Union[List[float], np.ndarray]
"""
The type of vectors, representing a list of coordinates in a high dimensional 
space.
"""
