# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Dict, Optional, Any, List
import uuid

from pymilvus import Collection, DataType, connections

from .vector_store import VectorStore
from ..common import Vector, Point


class MilvusVectorStore(VectorStore):
    """
    The vector store based on the Milvus vector database.
    """
    def __init__(self,
                 connection_args: Dict,
                 collection_name: str) -> None:
        super().__init__()
        # Connecting to Milvus instance
        if not connections.has_connection("default"):
            connections.connect(**connection_args)
        self._collection_name = collection_name
        self._collection = Collection(collection_name)
        self._auto_id = self._collection.schema.auto_id
        self._id_field = None
        self._vector_field = None
        self._fields = []
        # Grabbing the fields for the existing collection.
        for f in self._collection.schema.fields:
            self._fields.append(f.name)
            if f.is_primary and self._id_field is None:
                self._id_field = f.name
            elif f.dtype == DataType.FLOAT_VECTOR and self._vector_field is None:
                self._vector_field = f.name
        if self._id_field is None:
            raise ValueError(f"No primary field in the collection {collection_name}.")
        if self._vector_field is None:
            raise ValueError(f"No float vector field in the collection {collection_name}.")
        # load the collection
        self._collection.load()
        # Default search params when one is not provided.
        self._index_params = {
            "IVF_FLAT": {"params": {"nprobe": 10}},
            "IVF_SQ8": {"params": {"nprobe": 10}},
            "IVF_PQ": {"params": {"nprobe": 10}},
            "HNSW": {"params": {"ef": 10}},
            "RHNSW_FLAT": {"params": {"ef": 10}},
            "RHNSW_SQ": {"params": {"ef": 10}},
            "RHNSW_PQ": {"params": {"ef": 10}},
            "IVF_HNSW": {"params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"params": {"search_k": 10}},
        }

    def add(self,
            point: Point,
            **kwargs: Any) -> str:
        ids = self.add_all([point], **kwargs)
        return ids[0]

    def add_all(self,
                points: List[Point],
                **kwargs: Any) -> List[str]:
        data: List[List[Any]] = [] * len(self._fields)
        for p in points:
            if not self._auto_id and p.id is None:
                p.id = uuid.uuid4()
            for i, field in enumerate(self._fields):
                if field == self._id_field:
                    data[i].append(p.id)
                elif field == self._vector_field:
                    data[i].append(p.vector)
                else:
                    if p.metadata is None or field not in p.metadata.keys():
                        value = None
                    else:
                        value = p.metadata[field]
                    data[i].append(value)
        self._logger.debug("Insert data: %s", data)
        result = self._collection.insert(data=data, **kwargs)
        self._collection.flush()
        # set the automatically generated IDs for points
        if self._auto_id:
            for i, value in enumerate(result.primary_keys):
                points[i].id = value
        return [p.id for p in points]

    def search(self,
               vector: Vector,
               limit: int,
               filter: Optional[Any] = None,
               **kwargs: Any) -> List[Point]:
        pass

    def close(self) -> None:
        self._collection.release()
        connections.disconnect()
