# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from enum import Enum


class Protocol(Enum):
    """
    The enumeration of communication protocols.
    """
    HTTP = "http"

    HTTPS = "HTTPS"

    GRPC = "gRPC"
