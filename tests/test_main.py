import unittest

from main import Path
from main import Rule
from main import Sample
from pyLomma.main import Triple


class TripleTest(unittest.TestCase):

    def test__repr(self):
        for triple, expected in [
            (Triple("src", "rel", "tgt"), "rel(src,tgt)"),
            (Triple("X", "rel", "tgt"), "rel(X,tgt)"),
            (Triple("src", "rel", "Y"), "rel(src,Y)"),
            (Triple("X", "rel", "Y"), "rel(X,Y)"),
        ]:
            with self.subTest("Triple.__repr__", triple=triple, expected=expected):
                result = repr(triple)
                assert result == expected, f"{expected} expected, but {result} found"

    def test__convert(self):
        for value, subst, expected in [
            ("val", {}, "A2"),
            ("val", {"val": "X"}, "X"),
            ("val", {"val": "val"}, "val"),
            ("val", {"key": "X"}, "A2"),
            ("val", {"key": "X", "val": "Y"}, "Y"),
            ("val", {"key": "X", "val": "val"}, "val"),
        ]:
            with self.subTest("Triple.convert", value=value, subst=subst, expected=expected):
                result = Triple.convert(value, subst)
                assert result == expected, f"{expected} expected, but {result} found"

    def test__replace(self):
        for triple, subst, expected in [
            (Triple("max", "speaks", "english"), {}, Triple("A2", "speaks", "A3")),
            (Triple("max", "speaks", "english"), {"max": "X", "": ""}, Triple("X", "speaks", "A2")),
            (Triple("max", "speaks", "english"), {"english": "Y"}, Triple("A2", "speaks", "Y")),
            (Triple("max", "speaks", "english"), {"max": "X", "english": "Y"}, Triple("X", "speaks", "Y")),
            (Triple("max", "speaks", "english"), {"max": "max"}, Triple("max", "speaks", "A2")),
            (Triple("max", "speaks", "english"), {"english": "english"}, Triple("A2", "speaks", "english")),
            (Triple("max", "speaks", "english"), {"max": "max", "english": "Y"}, Triple("max", "speaks", "Y")),
            (Triple("max", "speaks", "english"), {"max": "X", "english": "english"}, Triple("X", "speaks", "english")),
            (Triple("max", "speaks", "english"), {"max": "max", "english": "english"},
             Triple("max", "speaks", "english")),
        ]:
            with self.subTest("Triple.replace", triple=triple, subst=subst, expected=expected):
                result = triple.replace(subst)
                assert result == expected, f"{expected} expected, but {result} found"


class SampleTest(unittest.TestCase):

    def test__repr(self):
        for sample, expected in [
            (Sample(Triple("src", "rel", "tgt"), False), "rel(src,tgt)"),
            (Sample(Triple("X", "rel", "tgt"), False), "rel(X,tgt)"),
            (Sample(Triple("src", "rel", "Y"), False), "rel(src,Y)"),
            (Sample(Triple("X", "rel", "Y"), False), "rel(X,Y)"),
            (Sample(Triple("src", "rel", "tgt"), True), "rel(src,tgt)"),
            (Sample(Triple("X", "rel", "tgt"), True), "rel(X,tgt)"),
            (Sample(Triple("src", "rel", "Y"), True), "rel(src,Y)"),
            (Sample(Triple("X", "rel", "Y"), True), "rel(X,Y)"),
        ]:
            with self.subTest("Sample.__repr__", triple=sample, expected=expected):
                result = repr(sample)
                assert result == expected, f"{expected} expected, but {result} found"

    def test__get_origin(self):
        for src, rel, tgt, inv, expected in [
            ("src", "rel", "tgt", False, "src"),
            ("src", "rel", "tgt", True, "tgt"),
        ]:
            with self.subTest("Sample.get_origin", src=src, rel=rel, tgt=tgt, inv=inv, expected=expected):
                result = Sample(Triple(src, rel, tgt), inv).get_origin()
                assert result == expected, f"{expected} expected, but {result} found"

    def test__get_destination(self):
        for src, rel, tgt, inv, expected in [
            ("src", "rel", "tgt", False, "tgt"),
            ("src", "rel", "tgt", True, "src"),
        ]:
            with self.subTest("Sample.get_origin", src=src, rel=rel, tgt=tgt, inv=inv, expected=expected):
                result = Sample(Triple(src, rel, tgt), inv).get_destination()
                assert result == expected, f"{expected} expected, but {result} found"


class PathTest(unittest.TestCase):

    def setUp(self):
        self.path1 = Path(
            Sample(Triple("max", "speaks", "english"), True),
            [Sample(Triple("max", "lives", "uk"), False), Sample(Triple("uk", "lang", "english"), False)],
            True
        )
        self.path2 = Path(
            Sample(Triple("max", "speaks", "english"), True),
            [Sample(Triple("max", "married", "eve"), False), Sample(Triple("eve", "born", "london"), False)],
            True
        )

    def test__len(self):
        for path, expected in [
            (self.path1, 2),
            (self.path2, 2),
        ]:
            with self.subTest("Path.__len__", path=path, expected=expected):
                result = len(path)
                assert result == expected, f"{expected} expected, but {result} found"

    def test__repr(self):
        for path, expected in [
            (self.path1, "speaks(max,english) :- lives(max,uk), lang(uk,english)."),
            (self.path2, "speaks(max,english) :- married(max,eve), born(eve,london)."),
        ]:
            with self.subTest("Path.__repr__", path=path, expected=expected):
                result = repr(path)
                assert result == expected, f"{expected} expected, but {result} found"

    def test__is_cyclic(self):
        for path, expected in [
            (self.path1, True),
            (self.path2, False),
        ]:
            with self.subTest("Path.is_cyclic", path=path, expected=expected):
                result = path.is_cyclic()
                assert result == expected, f"{expected} expected, but {result} found"

    def test__is_valid(self):
        for path, expected in [
            (self.path1, True),
            (self.path2, True),
        ]:
            with self.subTest("Path.is_valid", path=path, expected=expected):
                result = path.is_valid()
                assert result == expected, f"{expected} expected, but {result} found"

class RuleTest(unittest.TestCase):

    def setUp(self):
        self.path1 = Path(
            Sample(Triple("max", "speaks", "english"), True),
            [Sample(Triple("max", "lives", "uk"), False), Sample(Triple("uk", "lang", "english"), False)],
            True
        )
        self.path2 = Path(
            Sample(Triple("max", "speaks", "english"), True),
            [Sample(Triple("max", "married", "eve"), False), Sample(Triple("eve", "born", "london"), False)],
            True
        )

        self.rule1 = Rule(Triple("X", "speaks", "Y"), [Triple("X", "lives", "A2"), Triple("A2", "lang", "Y")])
        self.rule2 = Rule(Triple("X", "speaks", "english"), [Triple("X", "lives", "A2"), Triple("A2", "lang", "english")])
        self.rule3 = Rule(Triple("max", "speaks", "Y"), [Triple("max", "lives", "A2"), Triple("A2", "lang", "Y")])

        self.rule4 = Rule(Triple("X", "speaks", "english"), [Triple("X", "married", "A2"), Triple("A2", "born", "A3")])
        self.rule5 = Rule(Triple("X", "speaks", "english"), [Triple("X", "married", "A2"), Triple("A2", "born", "london")])


    def test__repr(self):
        for rule, expected in [
            (self.rule1, "speaks(X,Y) :- lives(X,A2), lang(A2,Y)."),
            (self.rule2, "speaks(X,english) :- lives(X,A2), lang(A2,english)."),
            (self.rule3, "speaks(max,Y) :- lives(max,A2), lang(A2,Y)."),
        ]:
            with self.subTest("Rule.__repr__", rule=rule, expected=expected):
                result = repr(rule)
                assert result == expected, f"{expected} expected, but {result} found"

    def test__generalization(self):
        for path, subst, expected in [
            (self.path1, {"max": "X", "english": "Y"}, self.rule1),
            (self.path1, {"max": "X", "english": "english"}, self.rule2),
            (self.path1, {"max": "max", "english": "Y"}, self.rule3),
        ]:
            with self.subTest("generalize", path=path, subst=subst, expected=expected):
                result = Rule.generalize(path, subst)
                assert result == expected, f"{expected} expeced, but {result} found"
