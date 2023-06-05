# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.common import Distance


class TestDistance(unittest.TestCase):

    def test_between(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        d1 = Distance.COSINE.between(v1, v2)
        print(f"d1={d1}")
        d2 = Distance.DOT.between(v1, v2)
        print(f"d2={d2}")
        d3 = Distance.EUCLID.between(v1, v2)
        print(f"d3={d3}")


if __name__ == '__main__':
    unittest.main()
