# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest
from unittest.mock import patch, mock_open
import threading
from requests import Session, exceptions

from llmsdk.util.common_utils import (
    global_init,
    read_config_file,
    is_website_accessible,
    extract_argument,
    record_to_csv,
    records_to_csv,
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


if __name__ == "__main__":
    unittest.main()
