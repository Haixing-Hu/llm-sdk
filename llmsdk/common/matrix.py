# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import List, Union, TypeAlias

import numpy as np


Matrix: TypeAlias = Union[List[List[float]], List[np.ndarray], np.ndarray]
"""
The type of 2-dimensional floating point matrix.
"""
