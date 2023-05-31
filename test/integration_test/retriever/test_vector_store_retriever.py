# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from qdrant_client import QdrantClient

from llmsdk.common import SearchType, Document, Metadata
from llmsdk.embedding import MockEmbedding
from llmsdk.vectorstore import QdrantVectorStore
from llmsdk.splitter import CharacterTextSplitter
from llmsdk.retriever import VectorStoreRetriever


class TestVectorStoreRetriever(unittest.TestCase):

    def test_open_close__memory(self):
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client)
        embedding = MockEmbedding()
        splitter = CharacterTextSplitter()
        collection_name = "FAQ"
        retriever = VectorStoreRetriever(vector_store=store,
                                         collection_name=collection_name,
                                         embedding=embedding,
                                         splitter=splitter,
                                         search_type=SearchType.SIMILARITY)
        self.assertEqual(False, retriever.is_opened)
        self.assertEqual(False, retriever.vector_store.is_opened)
        self.assertEqual(False, retriever.vector_store.is_collection_opened)
        self.assertEqual(False, retriever.vector_store.has_collection(collection_name))
        retriever.open()
        try:
            self.assertEqual(True, retriever.is_opened)
            self.assertEqual(True, retriever.vector_store.is_opened)
            self.assertEqual(True, retriever.vector_store.is_collection_opened)
            self.assertEqual(True, retriever.vector_store.has_collection(collection_name))
            self.assertEqual(collection_name, retriever.vector_store.collection_name)
        finally:
            retriever.close()
            self.assertEqual(False, retriever.is_opened)
            self.assertEqual(False, retriever.vector_store.is_opened)
            self.assertEqual(False, retriever.vector_store.is_collection_opened)
            self.assertEqual(True, retriever.vector_store.has_collection(collection_name))
            self.assertIsNone(retriever.vector_store.collection_name)

    def test_open_close__localhost(self):
        client = QdrantClient(location="127.0.0.1")
        store = QdrantVectorStore(client)
        embedding = MockEmbedding()
        splitter = CharacterTextSplitter()
        collection_name = "FAQ"
        retriever = VectorStoreRetriever(vector_store=store,
                                         collection_name=collection_name,
                                         embedding=embedding,
                                         splitter=splitter,
                                         search_type=SearchType.SIMILARITY)
        self.assertEqual(False, retriever.is_opened)
        self.assertEqual(False, retriever.vector_store.is_opened)
        self.assertEqual(False, retriever.vector_store.is_collection_opened)
        self.assertEqual(False, retriever.vector_store.has_collection(collection_name))
        retriever.open()
        try:
            self.assertEqual(True, retriever.is_opened)
            self.assertEqual(True, retriever.vector_store.is_opened)
            self.assertEqual(True, retriever.vector_store.is_collection_opened)
            self.assertEqual(True, retriever.vector_store.has_collection(collection_name))
            self.assertEqual(collection_name, retriever.vector_store.collection_name)
        finally:
            store.delete_collection(collection_name)
            retriever.close()
            self.assertEqual(False, retriever.is_opened)
            self.assertEqual(False, retriever.vector_store.is_opened)
            self.assertEqual(False, retriever.vector_store.is_collection_opened)
            self.assertEqual(False, retriever.vector_store.has_collection(collection_name))
            self.assertIsNone(retriever.vector_store.collection_name)

    def test_add__memory(self):
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client)
        embedding = MockEmbedding()
        splitter = CharacterTextSplitter()
        collection_name = "FAQ"
        retriever = VectorStoreRetriever(vector_store=store,
                                         collection_name=collection_name,
                                         embedding=embedding,
                                         splitter=splitter,
                                         search_type=SearchType.SIMILARITY)
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        retriever.open()
        try:
            retriever.add_all(documents)
            info = store.get_collection_info(collection_name)
            self.assertEqual(3, info.size)
        finally:
            store.delete_collection(collection_name)
            retriever.close()

    def test_add__localhost(self):
        client = QdrantClient(location="127.0.0.1")
        store = QdrantVectorStore(client)
        embedding = MockEmbedding()
        splitter = CharacterTextSplitter()
        collection_name = "FAQ"
        retriever = VectorStoreRetriever(vector_store=store,
                                         collection_name=collection_name,
                                         embedding=embedding,
                                         splitter=splitter,
                                         search_type=SearchType.SIMILARITY)
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        retriever.open()
        try:
            retriever.add_all(documents)
            info = store.get_collection_info(collection_name)
            self.assertEqual(3, info.size)
        finally:
            store.delete_collection(collection_name)
            retriever.close()

    def test_add_retrieve__memory(self):
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client)
        embedding = MockEmbedding()
        splitter = CharacterTextSplitter()
        collection_name = "FAQ"
        retriever = VectorStoreRetriever(vector_store=store,
                                         collection_name=collection_name,
                                         embedding=embedding,
                                         splitter=splitter,
                                         search_type=SearchType.SIMILARITY)
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        retriever.open()
        try:
            retriever.add_all(documents)
            result = retriever.retrieve("foo", limit=1)
            self.assertEqual(1, len(result))
            result[0].id = None
            self.assertEqual(documents[0], result[0])

            result = retriever.retrieve("foo", limit=2)
            self.assertEqual(2, len(result))
            result[0].id = None
            result[1].id = None
            self.assertEqual(documents[0], result[0])
            self.assertEqual(documents[1], result[1])
        finally:
            store.delete_collection(collection_name)
            retriever.close()


if __name__ == '__main__':
    unittest.main()
