# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Dict, Any, List, Iterable, Iterator, Callable
import threading
import requests

import csv
from io import StringIO
from tqdm import tqdm
from cachetools import Cache


def global_init(func):
    """
    A decorator that ensures a function is globally initialized only once, even
    in a multithread environment.

    :Examples:

    The following example demonstrates how to use the `global_init` decorator to
    ensure a global initialization function is only executed once, regardless of
    how many times it is called:

    .. code-block:: python

        @global_init
        def initialize():
            # Global initialization code here
            print("Global initialization")

        initialize()  # Only executed once
        # Output: Global initialization

        initialize()  # Not executed again

    :param func: The function to be decorated.
    :type func: callable
    :return: The decorated function.
    :rtype: callable
    """
    func.__lock__ = threading.Lock()
    func.__initialized__ = False

    def wrapper(*args, **kwargs):
        """
        Wrapper function that implements the double-checked locking mechanism to
        globally initialize a function.

        This wrapper function implements the double-checked locking mechanism to
        ensure that the decorated function is globally initialized only once. It
        first checks if the function has been initialized without acquiring
        the lock, and if not, it acquires the lock and performs the initialization.
        The use of the lock and the __initialized__ flag provides thread safety
        in a multithread environment.

        :param args: Positional arguments passed to the decorated function.
        :param kwargs: Keyword arguments passed to the decorated function.
        :return: The result of the decorated function.
        """
        if not func.__initialized__:
            with func.__lock__:
                if not func.__initialized__:
                    func(*args, **kwargs)
                    func.__initialized__ = True

    return wrapper


def read_config_file(file_path: str) -> Dict[str, str]:
    """
    Read the specified configuration file.

    The configuration file must be a text file. Each line contains a pair of a
    key and a value in the form of "key = value". A line starting with "#" is a
    comment line. Any line do not contain "=" symbol will be ignored.

    :param file_path: the path of the configuration file.
    :return: the key-value pairs read from the configuration file.
    """
    result = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # 去除行末尾的空格和换行符
            if not line.startswith('#') and '=' in line:  # 忽略注释行和没有=号的行
                key, value = line.split('=', 1)  # 将行拆分为键和值
                result[key.strip()] = value.strip()  # 将键值对添加到字典中
    return result


def is_website_accessible(url: str,
                          timeout: int = 3) -> bool:
    """
    Check if a website can be accessed by sending an HTTP GET request to the
    specified URL.

    :param url: The URL of the website to check.
    :param timeout: The number of seconds to wait for the server to respond.
    :return: True if the website can be accessed, False otherwise.

    :Example:

    The following example demonstrates how to use the `is_website_accessible`
    function to check if a website can be accessed:

    .. code-block:: python

        # Check if Google's website can be accessed
        url = "https://www.google.com"
        is_accessible = is_website_accessible(url, timeout=3)
        if is_accessible:
            print(f"Successfully connected to {url}")
        else:
            print(f"Failed to connect to {url}")
    """
    try:
        session = requests.Session()
        response = session.get(url, timeout=timeout, proxies={"no_proxy": "*"})
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def extract_argument(kwargs: Dict[str, Any], name: str, default_value: Any) -> Any:
    """
    Extract an argument from a dictionary.

    :param kwargs: the dictionary to be extracted. If the specified argument is
        found in this dictionary, after calling this function, the argument will
        be removed from this dictionary.
    :param name: the name of the argument to be extracted.
    :param default_value: the default value of the argument.
    :return: the value of the argument, or the default value if the argument is
        not found.
    """
    if name in kwargs:
        value = kwargs[name]
        del kwargs[name]
        return value
    else:
        return default_value


def record_to_csv(record: Dict[str, Any]) -> str:
    """
    Convert a record to a CSV string.

    :param record: the record to be converted.
    :return: the CSV string of the record.
    """
    header = []
    row = []
    for key, value in record.items():
        header.append(key)
        row.append(value)
    # Create a CSV file in memory
    csv_file = StringIO()
    csv_writer = csv.writer(csv_file, lineterminator='\n')
    # Write CSV data
    csv_writer.writerow(header)
    csv_writer.writerow(row)
    # Get CSV data
    return csv_file.getvalue()


def records_to_csv(records: List[Dict[str, Any]]) -> str:
    """
    Convert a list of records to a CSV string.

    :param records: the list of records to be converted.
    :return: the CSV string of the records.
    """
    header = []
    rows = []
    for record in records:
        for key in record.keys():
            if key not in header:
                header.append(key)
    for record in records:
        row = []
        for key in header:
            if key in record:
                row.append(record[key])
            else:
                row.append("")
        rows.append(row)
    # Create a CSV file in memory
    csv_file = StringIO()
    csv_writer = csv.writer(csv_file, lineterminator='\n')
    # Write CSV data
    csv_writer.writerow(header)
    csv_writer.writerows(rows)
    # Get CSV data
    return csv_file.getvalue()


def get_iterable_or_tqdm(iterable: Any,
                         show_progress: bool,
                         min_size_to_show: int) -> Any:
    """
    Get an iterable or a tqdm progress bar.

    :param iterable: the iterable to be processed.
    :param show_progress: whether to show the progress bar.
    :param min_size_to_show: the minimum size of the iterable to show the
        progress bar.
    :return: the iterable or the tqdm progress bar.
    """
    if show_progress and len(iterable) >= min_size_to_show:
        return tqdm(iterable)
    else:
        return iterable


def generate_uncached_items(items: Iterable[Any],
                            cache: Cache) -> Iterable[Any]:
    """
    Generates the items in an iterable that are not in the cache.

    :param items: the iterable to be processed.
    :param cache: the cache to be checked.
    :return: the generator of the items in the iterable that are not in
        the cache.
    """
    for item in items:
        if item not in cache:
            yield item


def generate_uncached_unique_items(items: Iterable[Any],
                                   cache: Cache) -> Iterable[Any]:
    """
    Generates the unique items in an iterable that are not in the cache.

    :param items: the iterable to be processed.
    :param cache: the cache to be checked.
    :return: the generator of the unique items in the iterable that are not in
        the cache.
    """
    unique_uncached_items = set()
    for item in items:
        print(f"generate_uncached_unique_items: item = {item}")
        if (item not in cache) and (item not in unique_uncached_items):
            unique_uncached_items.add(item)
            print(f"generate_uncached_unique_items: yield {item}")
            yield item


def generate_result_with_cache(
        items: Iterable[Any],
        cache: Cache,
        processor: Callable[[Iterable[Any]], Iterable[Any]]) -> Iterable[Any]:
    """
    Generates the processing results for the items in an iterable inputs, with
    the help of a cache to avoid processing duplicated items.

    :param items: the iterable items to be processed.
    :param cache: the cache to be checked.
    :param processor: the processor to be used to process the items in the
        iterable.
    :return: the generator of the processing results for the items in the iterable
        inputs.
    """
    class Input:
        def __init__(self, items_iter, cache, queue) -> None:
            self._items_iter = items_iter
            self._cache = cache
            self._queue = queue
            self._unique_set = set()

        def __iter__(self):
            index = 0
            while index < len(self._queue):
                val = self._queue[index]
                if val not in self._unique_set:
                    self._unique_set.add(val)
                    yield val
                index = index + 1

    items_iter = iter(items)
    queue = []
    input = Input(items_iter, cache, queue)
    output = processor(input)
    output_iter = iter(output)
    for item in items_iter:
        print(f"generate_result_with_cache: item = {item}")
        queue.append(item)
        while len(queue) > 0:
            print(f"generate_result_with_cache: queue = {queue}")
            top = queue[0]
            print(f"generate_result_with_cache: top = {top}")
            if top in cache:
                print(f"generate_result_with_cache: {top} in cache")
                value = cache[top]
            else:
                print(f"generate_result_with_cache: {top} NOT in cache, generate next result")
                value = next(output_iter)
                cache[top] = value
            print(f"generate_result_with_cache: yield {value}")
            queue.pop(0)
            yield value


def batch_generator(items: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """
    Generates batches of items from an iterable.

    :param items: the iterable items to be processed.
    :param batch_size: the size of each batch.
    :return: the generator of batches of items.
    """
    if batch_size <= 0:
        raise ValueError(f"The batch size must be positive: {batch_size}")
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
