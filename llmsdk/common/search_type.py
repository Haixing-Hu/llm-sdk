# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from enum import Enum


class SearchType(Enum):
    """
    The enumeration of searching types.
    """

    SIMILARITY = "similarity"

    MAX_MARGINAL_RELEVANCE = "max_marginal_relevance"
