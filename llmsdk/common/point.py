# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .vector import Vector
from .metadata import Metadata


@dataclass
class Point:
    """
    The class of points.

    The points are the central entity that a VectorStore operates with. A point
    is a record consisting of a vector, an optional ID, an optional metadata,
    and an optional score.
    """

    vector: Vector = field(default_factory=list)
    """The list of coordinates of a vector"""

    metadata: Metadata = field(default_factory=Metadata)
    """The metadata of this point."""

    id: Optional[str] = None
    """The ID of this point."""

    score: Optional[float] = None
    """The score of this point, which is set for searching result."""
