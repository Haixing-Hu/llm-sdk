# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import List, Optional, Tuple

import numpy as np

from ..common import Vector, Matrix


def matrix_cosine_similarity(x: Matrix, y: Matrix) -> np.ndarray:
    """
    Calculates the row-wise cosine similarity between two equal-width matrices.

    :param x: the first matrix.
    :param y: the second matrix, same width as x.
    :return: the row-wise cosine similarity between the two equal-width matrices.
    """
    if len(x) == 0 or len(y) == 0:
        return np.array([])
    x = np.array(x)
    y = np.array(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {x.shape} "
            f"and Y has shape {y.shape}."
        )
    x_norm = np.linalg.norm(x, axis=1)
    y_norm = np.linalg.norm(y, axis=1)
    similarity = np.dot(x, y.T) / np.outer(x_norm, y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def matrix_cosine_similarity_top_k(
    x: Matrix,
    y: Matrix,
    top_k: Optional[int] = 5,
    score_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Calculates the row-wise cosine similarity with optional top-k and score
    threshold filtering.

    :param x: the first matrix.
    :param y: the second matrix, same width as x.
    :param top_k: Max number of results to return.
    :param score_threshold: Minimum cosine similarity of results.
    :return: Tuple of two lists. First contains two-tuples of indices
        (x_index, y_index), second contains corresponding cosine similarities.
    """
    if len(x) == 0 or len(y) == 0:
        return [], []
    score_array = matrix_cosine_similarity(x, y)
    sorted_indices = score_array.flatten().argsort()[::-1]
    top_k = top_k or len(sorted_indices)
    top_indices = sorted_indices[:top_k]
    score_threshold = score_threshold or -1.0
    top_indices = top_indices[score_array.flatten()[top_indices] > score_threshold]
    return_indices = [(x // score_array.shape[1], x % score_array.shape[1])
                      for x in top_indices]
    scores = score_array.flatten()[top_indices].tolist()
    return return_indices, scores


def maximal_marginal_relevance(query_vector: Vector,
                               similarity_vectors: List[Vector],
                               lambda_multipy: float = 0.5,
                               limit: int = 4) -> List[int]:
    """
    Calculates the maximal marginal relevance.

    :param query_vector:
    :param similarity_vectors:
    :param lambda_multipy:
    :param limit:
    :return: the maximal marginal relevance.
    """
    if min(limit, len(similarity_vectors)) <= 0:
        return []
    query_vector = np.array(query_vector)
    if query_vector.ndim == 1:
        query_vector = np.expand_dims(query_vector, axis=0)
    similarity_to_query = matrix_cosine_similarity(query_vector, similarity_vectors)[0]
    most_similar = int(np.argmax(similarity_to_query))
    indices = [most_similar]
    selected = np.array([similarity_vectors[most_similar]])
    while len(indices) < min(limit, len(similarity_vectors)):
        best_score = -np.inf
        index_to_add = -1
        similarity_to_selected = matrix_cosine_similarity(similarity_vectors, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in indices:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = lambda_multipy * query_score \
                - (1 - lambda_multipy) * redundant_score
            if equation_score > best_score:
                best_score = equation_score
                index_to_add = i
        indices.append(index_to_add)
        selected = np.append(selected, [similarity_vectors[index_to_add]], axis=0)
    return indices
