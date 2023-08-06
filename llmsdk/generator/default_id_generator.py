# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import TypeAlias

from .uuid4_id_generator import Uuid4Generator


DefaultIdGenerator: TypeAlias = Uuid4Generator
"""
The class of default ID generator.
"""
