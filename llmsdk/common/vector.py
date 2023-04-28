# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List, Dict, Optional

from .point import Point
from .metadata import Metadata


class Vector:
    """
    The class of vectors.
    """
    def __init__(self,
                 point: Point,
                 id: Optional[str] = None,
                 score: Optional[float] = None,
                 metadata: Optional[Metadata] = None) -> None:
        """
        Construct a Vector object.

        :param point: the list of coordinates of this vector.
        :param id: the ID of this vector.
        :param score: the score of this vector, which is set for searching result.
        :param metadata: the metadata of this vector.
        """
        self._point = point
        self._id = id
        self._score = score
        self._metadata = {} if metadata is None else metadata

    @property
    def id(self) -> Optional[str]:
        return self._id

    @id.setter
    def id(self, value: Optional[str]) -> None:
        self._id = value

    @property
    def point(self) -> Point:
        return self._point

    @point.setter
    def point(self, value: Point) -> None:
        self._point = value

    @property
    def score(self) -> Optional[float]:
        return self._score

    @score.setter
    def score(self, value: Optional[float]) -> None:
        self._score = value

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Optional[Metadata]) -> None:
        self._metadata = {} if value is None else value

    def dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "point": self._point,
            "score": self._score,
            "metadata": self._metadata,
        }

    def __str__(self) -> str:
        return str(self.dict())

    def __repr__(self) -> str:
        return str(self)
