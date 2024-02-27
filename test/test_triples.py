import unittest

from pyLomma.main import Triple


class TripleTest(unittest.TestCase):

    def test_repr(self):
        triple = Triple("alpha", "beta", "gamma")
        assert repr(triple) == "beta(alpha,gamma)"
