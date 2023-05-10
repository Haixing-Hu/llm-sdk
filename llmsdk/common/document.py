# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass, field
from typing import Optional

from .metadata import Metadata


@dataclass
class Document:
    """
    The model represents documents.

    A document has a title, a content, and optional metadata.
    """

    content: str
    """The content of the document."""

    metadata: Optional[Metadata] = field(default_factory=dict)
    """The metadata of the document, or {} if no metadata."""
