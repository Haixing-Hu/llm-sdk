# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from collections import UserDict
from typing import Any, Dict, Type, Union


class Metadata(UserDict):
    """
    The class represents the metadata, which is a wrapper of a dict.
    """

    def __init__(self, data: Dict = None, **kwargs: Any):
        super().__init__(data, **kwargs)

    def __setitem__(self, key: str, value: Union[int, float, str]) -> None:
        if ((type(value) == int)
                or (type(value) == float)
                or (type(value) == str)):
            super().__setitem__(key, value)
        else:
            raise ValueError("The value of metadata only support int, float, "
                             "and str types.")

    def has_key(self, key: str, data_type: Type) -> bool:
        """
        Tests whether the metadata of this document has the specified attribute
        with the specified type.

        :param key: the name of the specified attribute.
        :param data_type: the data type of the specified attribute.
        :return: True if the metadata of this document has the specified attribute
            with the specified type；False otherwise.
        """
        return (key in self.data) and (type(self.data[key]) == data_type)
