# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from enum import Enum


class Protocol(Enum):
    """
    The enumeration of communication protocols.
    """
    HTTP = "http"

    HTTPS = "HTTPS"

    GRPC = "gRPC"
