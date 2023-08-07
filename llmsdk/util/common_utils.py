# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Dict, Any, List
import threading
import requests
import csv
from io import StringIO


def global_init(func):
    """
    A decorator that ensures a function is globally initialized only once, even
    in a multi-threaded environment.

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
        in a multi-threaded environment.

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
