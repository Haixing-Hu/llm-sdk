# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, Dict, Optional

from .vector import Vector
from .metadata import Metadata


class Point:
    """
    The class of points.

    The points are the central entity that a VectorStore operates with. A point
    is a record consisting of a vector, an optional ID, an optional metadata,
    and an optional score.
    """
    def __init__(self,
                 vector: Vector,
                 id: Optional[str] = None,
                 metadata: Optional[Metadata] = None,
                 score: Optional[float] = None) -> None:
        """
        Construct a Point object.

        :param vector: the list of coordinates of a vector.
        :param id: the ID of this point.
        :param metadata: the metadata of this point.
        :param score: the score of this point, which is set for searching result.
        """
        self._id = id
        self._vector = vector
        self._metadata = Metadata() if metadata is None else metadata
        self._score = score

    @property
    def id(self) -> Optional[str]:
        return self._id

    @id.setter
    def id(self, value: Optional[str]) -> None:
        self._id = value

    @property
    def vector(self) -> Vector:
        return self._vector

    @vector.setter
    def vector(self, value: Vector) -> None:
        self._vector = value

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Optional[Metadata]) -> None:
        self._metadata = {} if value is None else value

    @property
    def score(self) -> Optional[float]:
        return self._score

    @score.setter
    def score(self, value: Optional[float]) -> None:
        self._score = value

    def dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "vector": self._vector,
            "metadata": self._metadata,
            "score": self._score,
        }

    def __str__(self) -> str:
        return str(self.dict())

    def __repr__(self) -> str:
        return str(self)
