# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import uuid

from .id_generator import IdGenerator


class Uuid1Generator(IdGenerator):
    """
    The ID generator generating UUID1 IDs.
    """

    def generate(self) -> str:
        return str(uuid.uuid1())
