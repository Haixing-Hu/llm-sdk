# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Dict, Any
import threading
import requests


def singleton(func):
    """
    A decorator that creates a singleton instance of the decorated function.

    :param func: The function to be decorated.
    :type func: function
    :return: A new function that ensures only one instance of `func` is created.
    :rtype: function

    :Example:

    The following example demonstrates how to use the `@singleton` decorator to
    create a singleton instance of a global function called `my_global_function`:

    .. code-block:: python

        @singleton
        def my_global_function():
            # Your code here
            pass

        # Call the global function
        result = my_global_function()
    """
    func.__lock__ = threading.Lock()
    func.__instance__ = None

    def wrapper(*args, **kwargs):
        if not func.__instance__:
            with func.__lock__:
                if not func.__instance__:
                    func.__instance__ = func(*args, **kwargs)
        return func.__instance__
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
