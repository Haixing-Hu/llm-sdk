# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Optional

from cachetools import Cache, LRUCache


class WithCacheMixin:

    DEFAULT_CACHE_SIZE: int = 10000

    """
    A mixin class that provides the cache function.
    """
    def __init__(self, *,
                 use_cache: bool = True,
                 cache_size: int = DEFAULT_CACHE_SIZE) -> None:
        """
        Creates a new WithCacheMixin object.

        :param use_cache: indicates whether to use the cache.
        :param cache_size: the number of items to be cached. This argument is
            ignored if the `use_cache` argument is False.
        """
        super().__init__()      # This MUST be called
        self._use_cache = use_cache
        self._cache_size = cache_size
        self._cache = None
        self.set_cache(use_cache, cache_size)

    @property
    def use_cache(self) -> bool:
        return self._use_cache

    @property
    def cache_size(self) -> int:
        return self._cache_size

    @property
    def cache(self) -> Optional[Cache]:
        return self._cache

    def set_cache(self,
                  use_cache: bool,
                  cache_size: int = DEFAULT_CACHE_SIZE) -> None:
        """
        Sets the caching capacity of this object.

        :param use_cache: indicates whether to use the cache.
        :param cache_size: the number of items to be cached. This argument is
            ignored if the `use_cache` argument is False.
        """
        if cache_size <= 0:
            raise ValueError("The cache size must be positive.")
        self._use_cache = use_cache
        self._cache_size = cache_size
        if use_cache:
            self._cache = LRUCache(maxsize=cache_size)
        else:
            self._cache = None
