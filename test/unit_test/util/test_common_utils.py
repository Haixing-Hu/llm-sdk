# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Iterable
import unittest
from unittest.mock import patch, mock_open
import threading

from cachetools import LRUCache
from requests import Session, exceptions

from llmsdk.util.common_utils import (
    global_init,
    read_config_file,
    is_website_accessible,
    extract_argument,
    record_to_csv,
    records_to_csv,
    generate_uncached_items,
    generate_uncached_unique_items,
    generate_result_with_cache,
    batch_generator,
)


class TestCommonUtils(unittest.TestCase):
    def test_global_init_decorator_single_thread(self):
        init_count = 0

        @global_init
        def initialize():
            nonlocal init_count
            init_count += 1

        initialize()
        initialize()
        self.assertEqual(init_count, 1)

    def test_global_init_decorator_multi_threading(self):
        init_count = 0

        @global_init
        def initialize():
            nonlocal init_count
            init_count += 1

        num_threads = 10
        threads = []

        def call_initialize():
            initialize()

        for _ in range(num_threads):
            thread = threading.Thread(target=call_initialize)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(init_count, 1)

    def test_read_config_file(self):
        config_data = """
            # Sample Configuration File
            key1 = value1  
              key2=value2  
            key3 = value3
            key4 = 
            key5 value5
            key6 = value6 = extra
        """
        expected_result = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
            "key4": "",
            "key6": "value6 = extra"
        }

        with patch("builtins.open", mock_open(read_data=config_data)):
            result = read_config_file("config.txt")
        self.assertEqual(result, expected_result)

    def test_accessible_website(self):
        url = "https://www.baidu.com"
        with patch.object(Session, "get") as mock_get:
            mock_get.return_value.status_code = 200
            result = is_website_accessible(url)
        self.assertTrue(result)

    def test_inaccessible_website(self):
        url = "https://www.example.com"
        with patch.object(Session, "get") as mock_get:
            mock_get.side_effect = exceptions.RequestException
            result = is_website_accessible(url)
        self.assertFalse(result)

    def test_extract_existing_argument(self):
        kwargs = {"arg1": 10, "arg2": "value", "arg3": True}
        name = "arg2"
        default_value = None
        expected_result = "value"
        result = extract_argument(kwargs, name, default_value)
        self.assertEqual(result, expected_result)
        self.assertNotIn(name, kwargs)

    def test_extract_non_existing_argument(self):
        kwargs = {"arg1": 10, "arg2": "value", "arg3": True}
        name = "arg4"
        default_value = "default"
        expected_result = "default"
        result = extract_argument(kwargs, name, default_value)
        self.assertEqual(result, expected_result)
        self.assertNotIn(name, kwargs)

    def test_record_to_csv(self):
        record1 = {
            "id": "001",
            "name": "name1",
            "spec": "10ml",
            "value": 1,
            "delta": 0.13,
        }
        csv1 = record_to_csv(record1)
        expected_csv1 = ("id,name,spec,value,delta\n"
                         "001,name1,10ml,1,0.13\n")
        self.assertEqual(csv1, expected_csv1)

        record2 = {
            "id": "002",
            "name": "name,2\"",
            "spec": "20ml",
            "value": 2,
            "delta": 3.14,
        }
        csv2 = record_to_csv(record2)
        expected_csv2 = ("id,name,spec,value,delta\n"
                         "002,\"name,2\"\"\",20ml,2,3.14\n")
        self.assertEqual(csv2, expected_csv2)

    def test_records_to_csv(self):
        record1 = {
            "id": "001",
            "name": "name1",
            "spec": "10ml",
            "value": 1,
            "delta": 0.13,
        }
        record2 = {
            "id": "002",
            "name": "name,2\"",
            "brand": "qubit",
            "spec": "20ml",
            "value": 2,
            "delta": 3.14,
        }
        csv = records_to_csv([record1, record2])
        expected_csv = ("id,name,spec,value,delta,brand\n"
                        "001,name1,10ml,1,0.13,\n"
                        "002,\"name,2\"\"\",20ml,2,3.14,qubit\n")
        self.assertEqual(csv, expected_csv)

    def test_batch_generator(self):
        items = iter([])
        result = batch_generator(items, 3)
        result_iter = iter(result)
        self.assertRaises(StopIteration, next, result_iter)

        items = iter([1])
        result = batch_generator(items, 3)
        result_iter = iter(result)
        self.assertEqual([1], next(result_iter))
        self.assertRaises(StopIteration, next, result_iter)

        items = iter([1, 2])
        result = batch_generator(items, 3)
        result_iter = iter(result)
        self.assertEqual([1, 2], next(result_iter))
        self.assertRaises(StopIteration, next, result_iter)

        items = iter([1, 2, 3])
        result = batch_generator(items, 3)
        result_iter = iter(result)
        self.assertEqual([1, 2, 3], next(result_iter))
        self.assertRaises(StopIteration, next, result_iter)

        items = iter([1, 2, 2, 3, 1, 3, 4, 1, 5, 4, 4, 4, 5, 2, 3, 4, 1])
        result = batch_generator(items, 3)
        result_iter = iter(result)
        self.assertEqual([1, 2, 2], next(result_iter))
        self.assertEqual([3, 1, 3], next(result_iter))
        self.assertEqual([4, 1, 5], next(result_iter))
        self.assertEqual([4, 4, 4], next(result_iter))
        self.assertEqual([5, 2, 3], next(result_iter))
        self.assertEqual([4, 1], next(result_iter))
        self.assertRaises(StopIteration, next, result_iter)

        items = iter([1, 2, 3])
        result = batch_generator(items, -1)
        result_iter = iter(result)
        self.assertRaises(ValueError, next, result_iter)

    def test_generate_uncached_unique_items(self):
        values = [1, 2, 2, 3, 1, 3, 4, 1, 5, 4, 4, 4, 5, 2, 3]
        items = iter(values)
        cache = LRUCache(maxsize=5)
        cache[1] = "item1"
        cache[3] = "item3"
        result = generate_uncached_unique_items(items, cache)
        result_iter = iter(result)
        self.assertEqual(2, next(result_iter))
        self.assertEqual(4, next(result_iter))
        self.assertEqual(5, next(result_iter))
        self.assertRaises(StopIteration, next, result_iter)

        items = iter(values)
        result = generate_uncached_unique_items(items, cache)
        result_iter = iter(result)
        self.assertEqual(2, next(result_iter))
        cache[2] = "item2"
        self.assertEqual(4, next(result_iter))
        cache[4] = "item4"
        self.assertEqual(5, next(result_iter))
        cache[5] = "item5"
        self.assertRaises(StopIteration, next, result_iter)

    def test_generate_uncached_items(self):
        values = [1, 2, 2, 3, 1, 3, 4, 1, 5, 4, 4, 4, 5, 2, 3]
        items = iter(values)
        cache = LRUCache(maxsize=5)
        cache[1] = "item1"
        cache[3] = "item3"

        result = generate_uncached_items(items, cache)
        result_iter = iter(result)
        self.assertEqual(2, next(result_iter))
        self.assertEqual(2, next(result_iter))
        self.assertEqual(4, next(result_iter))
        self.assertEqual(5, next(result_iter))
        self.assertEqual(4, next(result_iter))
        self.assertEqual(4, next(result_iter))
        self.assertEqual(4, next(result_iter))
        self.assertEqual(5, next(result_iter))
        self.assertEqual(2, next(result_iter))
        self.assertRaises(StopIteration, next, result_iter)

        items = iter(values)
        result = generate_uncached_items(items, cache)
        result_iter = iter(result)
        self.assertEqual(2, next(result_iter))
        cache[2] = "item2"
        self.assertEqual(4, next(result_iter))
        cache[4] = "item4"
        self.assertEqual(5, next(result_iter))
        cache[5] = "item5"
        self.assertRaises(StopIteration, next, result_iter)

    def test_generate_result_with_cache__simple_loop_processor(self):
        def processor(ig: Iterable[int]) -> Iterable[str]:
            for i in ig:
                nonlocal counter
                counter = counter + 1
                r = f"new-item{i}-{counter}"
                print(f"processor: i = {i}, counter = {counter}, result = {r}")
                yield r

        counter = 0
        values = [1, 2, 4, 2, 2, 1, 3, 4, 1, 5, 4, 4, 4, 5, 2, 3]
        items = iter(values)
        cache = LRUCache(maxsize=5)
        cache[1] = "old-item1"
        cache[3] = "old-item3"
        result = generate_result_with_cache(items, cache, processor)
        result_iter = iter(result)
        self.assertEqual("old-item1", next(result_iter))
        self.assertEqual("new-item2-1", next(result_iter))
        self.assertEqual("new-item4-2", next(result_iter))
        self.assertEqual("new-item2-1", next(result_iter))
        self.assertEqual("new-item2-1", next(result_iter))
        self.assertEqual("old-item1", next(result_iter))
        self.assertEqual("old-item3", next(result_iter))
        self.assertEqual("new-item4-2", next(result_iter))
        self.assertEqual("old-item1", next(result_iter))
        self.assertEqual("new-item5-3", next(result_iter))
        self.assertEqual("new-item4-2", next(result_iter))
        self.assertEqual("new-item4-2", next(result_iter))
        self.assertEqual("new-item4-2", next(result_iter))
        self.assertEqual("new-item5-3", next(result_iter))
        self.assertEqual("new-item2-1", next(result_iter))
        self.assertEqual("old-item3", next(result_iter))
        self.assertRaises(StopIteration, next, result_iter)
        self.assertEquals("old-item1", cache[1])
        self.assertEquals("new-item2-1", cache[2])
        self.assertEquals("old-item3", cache[3])
        self.assertEquals("new-item4-2", cache[4])
        self.assertEquals("new-item5-3", cache[5])

    def test_generate_result_with_cache__batch_processor(self):
        def processor(ig: Iterable[int]) -> Iterable[str]:
            bg = batch_generator(ig, 3)
            for b in bg:
                for i in b:
                    nonlocal counter
                    counter = counter + 1
                    r = f"new-item{i}-{counter}"
                    print(f"processor: i = {i}, counter = {counter}, result = {r}")
                    yield r

        counter = 0
        values = [1, 2, 4, 2, 2, 1, 3, 4, 1, 5, 4, 4, 4, 5, 2, 3]
        items = iter(values)
        cache = LRUCache(maxsize=5)
        cache[1] = "old-item1"
        cache[3] = "old-item3"
        result = generate_result_with_cache(items, cache, processor)
        result_iter = iter(result)
        self.assertEqual("old-item1", next(result_iter))
        self.assertEqual("new-item2-1", next(result_iter))
        self.assertEqual("new-item4-2", next(result_iter))
        self.assertEqual("new-item2-1", next(result_iter))
        self.assertEqual("new-item2-1", next(result_iter))
        self.assertEqual("old-item1", next(result_iter))
        self.assertEqual("old-item3", next(result_iter))
        self.assertEqual("new-item4-2", next(result_iter))
        self.assertEqual("old-item1", next(result_iter))
        self.assertEqual("new-item5-3", next(result_iter))
        self.assertEqual("new-item4-2", next(result_iter))
        self.assertEqual("new-item4-2", next(result_iter))
        self.assertEqual("new-item4-2", next(result_iter))
        self.assertEqual("new-item5-3", next(result_iter))
        self.assertEqual("new-item2-1", next(result_iter))
        self.assertEqual("old-item3", next(result_iter))
        self.assertRaises(StopIteration, next, result_iter)
        self.assertEquals("old-item1", cache[1])
        self.assertEquals("new-item2-1", cache[2])
        self.assertEquals("old-item3", cache[3])
        self.assertEquals("new-item4-2", cache[4])
        self.assertEquals("new-item5-3", cache[5])


if __name__ == "__main__":
    unittest.main()
