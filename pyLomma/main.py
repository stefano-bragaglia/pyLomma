from __future__ import annotations

import argparse as ap
import csv
import logging
import multiprocessing as mp
import os
import re
import sys
from math import prod
from random import choice
from random import seed
from time import time as now
from typing import Generator
from typing import NamedTuple
from typing import Sequence
from typing import TextIO


def is_constant(value: str) -> bool:
    return value[0].islower()


class Parser:
    """ Naive implementation of a PEG parser.
    """

    def __init__(self):
        self.pos = 0
        self.content = None

    def expect(self, pattern) -> str | None:
        content = self.content[self.pos:]
        found = re.match(r"\s*", content)
        if found:
            self.pos += len(found.group(0))

        if pattern:
            content = self.content[self.pos:]
            found = re.match(pattern, content)
            if found:
                result = found.group(0)
                self.pos += len(result)
                return result

        return None

    def mark(self) -> int:
        return self.pos

    def reset(self, pos: int) -> None:
        self.pos = pos

    def parse(self, content: str) -> Path:
        self.pos = 0
        self.content = content

        result = self.start()
        self.expect(None)
        assert self.pos == len(self.content)

        return result

    def start(self) -> any:
        raise NotImplementedError()


class PathParser(Parser):
    """ Naive implementation of a PEG parser for Paths (no dependencies).
    """

    CONSTANT = r"[_a-z][a-zA-Z0-9_-]*"
    NUMBER = r"[0-9]+"
    VARIABLE = r"[A-Z][a-zA-Z0-9_-]*"

    def start(self):
        pos = self.mark()
        if head := self.triple():
            if self.expect(':') and (num := self.expect(self.NUMBER)):
                if self.expect('/') and (den := self.expect(self.NUMBER)):
                    if self.expect(':-') and (body := self.body()):
                        if self.expect(r'\.'):
                            return Path(head, eval(num), eval(den), body)

        self.reset(pos)
        return None

    def body(self):
        pos = self.mark()
        if triple := self.triple():
            result = [triple]
            while True:
                pos = self.mark()
                if not self.expect(',') or not (triple := self.triple()):
                    self.reset(pos)
                    break
                result.append(triple)
            return result

        self.reset(pos)
        return None

    def triple(self) -> Triple | None:
        pos = self.mark()
        if (rel := self.expect(self.CONSTANT)) and self.expect(r'\('):
            if (src := self.literal()) and self.expect(','):
                if (tgt := self.literal()) and self.expect(r'\)'):
                    return Triple(src, rel, tgt)

        self.reset(pos)
        return None

    def literal(self):
        return self.expect(self.CONSTANT) or self.expect(self.VARIABLE)


class Triple(NamedTuple):
    """ A (semantic) triple contains a subject or 'source', a predicate or 'relation', and an object or 'destination'.
    """

    source: str
    relation: str
    target: str

    def __repr__(self) -> str:
        return f"{self.relation}({self.source},{self.target})"

    @staticmethod
    def convert(value: str, subst: dict[str, str]) -> str:
        return subst.setdefault(value, f"A{2 + sum(1 for x in subst.values() if x.startswith("A"))}")

    def replace(self, subst: dict[str, str]) -> Triple:
        return Triple(self.convert(self.source, subst), self.relation, self.convert(self.target, subst))

    def subst(self, subst: dict[str, str]) -> Triple:
        return Triple(subst.get(self.source, self.source), self.relation, subst.get(self.target, self.target))


class Sample(NamedTuple):
    """ A sample is a triple randomly selected during a random walk.

    Inverse is a flag indicating whether the triple has been traversed according
    its natural direction or not.
    """

    triple: Triple
    inverse: bool

    def __hash__(self) -> int:
        return hash(self.triple) * hash(self.inverse)

    def __repr__(self) -> str:
        return repr(self.triple)

    def get_origin(self) -> str:
        """ Return the origin of the sample.

        The origin is either the source or the target of the triple depending
        on the value of the inverse flag.

        :return: the origin of the sample
        """
        return self.triple.target if self.inverse else self.triple.source

    def get_destination(self) -> str:
        """ Return the destination of the sample.

        The destination is either the target or the source of the triple depending
        on the value of the inverse flag.

        :return: the destination of the sample
        """
        return self.triple.source if self.inverse else self.triple.target

    def replace(self, subst: dict[str, str]) -> Sample:
        return Sample(self.triple.replace(subst), self.inverse)


class Path(NamedTuple):
    """ A path is a sequence of consecutive triples.
    """
    head: Triple
    correct_samples: int
    total_samples: int
    body: list[Triple]

    def __hash__(self) -> int:
        result = hash(self.head)
        for atom in self.body:
            result *= hash(atom)

        return result

    def __len__(self) -> int:
        return len(self.body)

    def __repr__(self):
        if not self.body:
            return f"{repr(self.head)} : {self.correct_samples} / {self.total_samples}."

        body = ", ".join(repr(a) for a in self.body)

        return f"{repr(self.head)} : {self.correct_samples} / {self.total_samples} :- {body}."

    @property
    def confidence(self) -> float:
        return self.correct_samples / self.total_samples

    def infer(self, index: Index, k: list[Triple] = None, s: dict[str, str] = None) -> Generator[Path, None, None]:
        k = k or []
        if len(k) == len(self.body):
            yield Path(self.head.subst(s), self.correct_samples, self.total_samples, k)
        else:
            s = s or {}
            atom = self.body[len(k)].subst(s)
            for triple in index.match(atom):
                if atom.source[0].isupper():
                    s.setdefault(atom.source, triple.source)
                if atom.target[0].isupper():
                    s.setdefault(atom.target, triple.target)
                yield from self.infer(index, [*k, triple], s)


class RandomWalk(NamedTuple):
    """ A random walk is a sequence of consecutive sampled triples (samples).
    A random walk can be generalized in rules of type C, AC1, and AC2.
    """

    head: Sample
    body: list[Sample]
    expected: bool

    def __hash__(self) -> int:
        result = hash(self.head)
        for atom in self.body:
            result *= hash(atom)
        result *= hash(self.expected)

        return result

    def __len__(self) -> int:
        return len(self.body)

    def __repr__(self):
        if not self.body:
            return f"{repr(self.head)}."

        return f"{repr(self.head)} :- {", ".join(repr(a) for a in self.body)}."

    def _c_subst(self) -> dict[str, str]:
        return {
            self.head.get_origin(): "Y",
            self.head.get_destination(): "X",
        }

    def _ac1x_subst(self) -> dict[str, str]:
        return {
            self.head.get_origin(): self.head.get_origin(),
            self.head.get_destination(): "X",
        }

    def _ac1y_subst(self) -> dict[str, str]:
        return {
            self.head.get_origin(): "Y",
            self.head.get_destination(): self.head.get_destination(),
        }

    def _ac1_subst(self) -> dict[str, str]:
        return {
            self.head.get_origin(): self.head.get_origin(),
            self.head.get_destination(): "X",
            self.body[-1].get_destination(): self.body[-1].get_destination(),
        }

    def _ac2_subst(self) -> dict[str, str]:
        return {
            self.head.get_origin(): self.head.get_origin(),
            self.head.get_destination(): "X",
        }

    def generalize(self) -> Generator[Rule, None, None]:
        if self.is_cyclic():
            yield Rule.create(self, self._c_subst())
            yield Rule.create(self, self._ac1x_subst())
            yield Rule.create(self, self._ac1y_subst())
        else:
            yield Rule.create(self, self._ac1_subst())
            yield Rule.create(self, self._ac2_subst())

    def is_cyclic(self) -> bool:
        """ Check if the path is cyclic.

        A path is cyclic if the origin of its head (c_0) is equal to the destination of the last body atom (c_n+1).

        :return: True if the path is cyclic, False otherwise
        """
        return self.body and self.head.get_origin() == self.body[-1].get_destination()

    def is_valid(self) -> bool:
        """ Check if the path is valid.

        A path is valid if all its nodes are traversed exactly once and the origin only
        appears as destination of the last body atom (if the path is cyclic).

        :return: True if the path is valid, False otherwise
        """
        terminal = self.head.get_origin()
        visited = [self.head.get_destination()]
        for i, atom in enumerate(self.body, start=1):
            node = atom.get_destination()
            if node == terminal and i < len(self.body) or node in visited:
                return False

            visited.append(node)

        return True


class Rule(NamedTuple):
    """ A rule is a sequence of samples with at least 1 variable.

    A lowercase string represent a constant; an uppercase string represent a variable.
    """

    head: Triple
    body: list[Triple]

    # The confidence of a rule is usually defined as number of body groundings,
    # divided by the number of those body groundings that make the head true.

    # THIS IS FOR COMBINING THE CONFIDENCES OF THE RULES WITH SAME HEAD (JUST VARS!!!)
    # Note that we count in terms of head groundings, e.g., we count the number
    # of different ⟨X, Y⟩ groundings with respect to Rule 6 and the number of
    # different X groundings with respect to Rule 7.

    # Several different STRATEGIES can be used to combine the confidence!!! Check back later on the paper!

    def __hash__(self) -> int:
        result = hash(self.head)
        for atom in self.body:
            result *= hash(atom)

        return result

    def __len__(self) -> int:
        return len(self.body)

    def __repr__(self):
        if not self.body:
            return f"{repr(self.head)}."

        return f"{repr(self.head)} :- {", ".join(repr(a) for a in self.body)}."

    @staticmethod
    def create(path: RandomWalk, subst: dict[str, str] = None) -> Rule:
        return Rule(path.head.triple.replace(subst), [a.triple.replace(subst) for a in path.body])

    # def replace(self, subst: dict[str, str]) -> RandomWalk:
    #     return RandomWalk(
    #         self.head.replace(subst),
    #         [atom.replace(subst) for atom in self.body]
    #     )


class KnowledgeGraph(NamedTuple):
    """ A knowledge graph is a collection of facts represented by (semantic) triples.
    """

    facts: list[Triple]

    @staticmethod
    def load(fp: TextIO) -> KnowledgeGraph:
        """ Load a knowledge graph from a file-like object.

        The content of the file-like object is expected to be a CSV file with fields:
        source, relation, and target.

        :param fp: a file-like object
        :return: the knowledge graph
        """
        start = now()
        reader = csv.DictReader(fp)
        assert reader.fieldnames == ["source", "relation", "target"], \
            "Expected a .csv file with fields: 'source', 'relation', and 'target'"

        facts = [Triple(**t) for t in reader]
        result = KnowledgeGraph(facts)
        logging.info(f"  {len(facts):,} triple/s loaded in {now() - start}s")

        return result

    def add(self, fact: Triple) -> None:
        """ Add a fact to the knowledge graph.

        :param fact: the (semantic) triple to add to the knowledge graph
        """
        self.facts.append(fact)

    def dump(self, fp: TextIO) -> None:
        """ Dump the knowledge graph to a file-like object.

        The content of the file-like object is expected to be a CSV file with fields:
        source, relation, and target.

        :param fp: a file-like object
        """
        start = now()
        facts = [t._asdict() for t in sorted(set(self.facts), key=lambda x: (x.relation, x.source, x.target))]
        writer = csv.DictWriter(fp, fieldnames=["source", "relation", "target"])
        writer.writeheader()
        writer.writerows(facts)
        logging.info(f"  {len(facts):,} triple/s saved in {now() - start}s")


class Index(NamedTuple):
    """ An index is a collection of triples indexed by source, relation, and target.
    """

    triples: list[Triple]
    by_source: dict[str, list[Triple]]
    by_relation: dict[str, list[Triple]]
    by_target: dict[str, list[Triple]]
    # sources_by_relation: dict[str, list[str]]
    # targets_by_relation: dict[str, list[str]]

    by_relation_source: dict[str, dict[str, list[Triple]]]
    by_relation_target: dict[str, dict[str, list[Triple]]]

    @staticmethod
    def populate(kg: KnowledgeGraph) -> Index:
        """ Populates an index from a knowledge graph.

        :param kg: the knowledge graph to index
        :return: the index on the knowledge graph
        """
        start = now()
        result = Index([], {}, {}, {}, {}, {})
        for triple in kg.facts:
            if triple not in result.triples:
                result.by_source.setdefault(triple.source, []).append(triple)
                result.by_relation.setdefault(triple.relation, []).append(triple)
                result.by_target.setdefault(triple.target, []).append(triple)
                result.by_relation_source.setdefault(triple.relation, {}).setdefault(triple.source, []).append(triple)
                result.by_relation_target.setdefault(triple.relation, {}).setdefault(triple.target, []).append(triple)
                result.triples.append(triple)

        logging.info(f"  {len(result.triples):,} unique triple/s ("
                     f"{len(result.by_source):,} source/s, "
                     f"{len(result.by_relation):,} relation/s, "
                     f"{len(result.by_target):,} target/s"
                     f") indexed in {now() - start}s")

        return result

    def sample(self, relation: str = None) -> tuple[Sample, bool]:
        """ Sample a triple from the training set and return it.

        :param relation: the desired relation
        :return: the sampled triple and a boolean flag telling if expected
        """
        if relation:
            sources = list(self.by_relation_source.get(relation, {}).keys())
            relations = []
            targets = list(self.by_relation_target.get(relation, {}).keys())
        else:
            sources = list(self.by_source)
            relations = list(self.by_relation)
            targets = list(self.by_target)
        if not sources or not targets or not relation and not relations:
            raise ValueError("No example/s found")

        triple = Triple(choice(sources), relation or choice(relations), choice(targets))
        inverse = choice([False, True])
        expected = triple in self.by_relation.get(relation, self.triples)

        return Sample(triple, inverse), expected

    def next_step(self, path: RandomWalk) -> Sample | None:
        """
        Return a possible next step of a random walk for the path.

        :param path: the path to extend
        :return: the next step or None if no next step is possible
        """
        if not path.body:
            node = path.head.get_destination()
        else:
            node = path.body[-1].get_destination()

        inverse = choice([False, True])
        if inverse:
            triples = self.by_target.get(node, [])
        else:
            triples = self.by_source.get(node, [])

        if not triples:
            return None

        return Sample(choice(triples), inverse)

    def random_walk(self, length: int, relation: str = None) -> RandomWalk | None:
        """ Return a path by performing a random walk from an example.

        :param length: the desired length for the path (min 2)
        :param relation: the optional relation to learn (if missing, learn any relation)
        :return: a sampled path or None
        """
        head, expected = self.sample(relation)
        path = RandomWalk(head, [], expected)
        while len(path) < length:
            atom = self.next_step(path)
            if not atom:
                return None

            path.body.append(atom)
            if not path.is_valid():
                return None

        return path

    def match(self, atom: Triple) -> Generator[Triple, None, None]:
        if is_constant(atom.source):
            triples = self.by_relation_source.get(atom.relation, {}).get(atom.source, [])
            if is_constant(atom.target):
                triples = [atom] if atom in triples else []
        elif is_constant(atom.target):
            triples = self.by_relation_target.get(atom.relation, {}).get(atom.target, [])
        else:
            triples = self.by_relation.get(atom.relation, [])

        for triple in triples:
            yield triple

    # def query(self, rule: Rule, subst: dict[str, str] = None, pos: int = None) -> Generator[RandomWalk, None, None]:
    #     pos = pos or 0
    #     if pos >= len(rule):
    #         yield rule.replace(subst)
    #     else:
    #         subst = subst or {}
    #         for triple in self.find(rule.body[pos].triple):
    #             triple = triple.replace(subst)
    #             # if pos <= 0 or rule.body[pos - 1].get_destination() == rule.body[pos].get_origin():
    #             subst.update(rule.body[pos].triple.get_subst(triple))
    #             yield from self.query(rule, subst, pos + 1)


class Codex:

    @staticmethod
    def discover(num: int, index: Index, length: int, args: ap.Namespace) -> Codex:
        rules = Codex()
        deadline = now() + args.duration
        while now() <= deadline:
            path = index.random_walk(length, args.relation)
            rules.add(path)

        filename = f"rules_{num}.pl"
        with open(filename, "w") as file:
            logging.debug(f"Saving {len(rules)} rule/s to '{filename}'...")
            for rule in rules.export():
                print(rule, file=file)

        return rules.filter(args.quality)

    def __init__(self, rules: dict[Rule, list[RandomWalk]] = None):
        self.rules = rules or {}

    def __len__(self) -> int:
        return len(self.rules)

    def add(self, path: RandomWalk) -> None:
        if path:
            for rule in path.generalize():
                paths = self.rules.setdefault(rule, [])
                if path not in paths:
                    paths.append(path)

    @staticmethod
    def count(paths: list[RandomWalk]) -> int:
        return sum(1 for p in paths if p.expected)

    def filter(self, min_quality: float) -> Codex:
        quality_rules = {r: l for r, l in self.rules.items() if self.count(l) / len(l) >= min_quality}

        return Codex(quality_rules)

    def get_saturation(self, result: Codex) -> float:
        if not result.rules:
            return 0.0

        return sum(1 for r in result.rules if r in self.rules) / len(result.rules)

    def update(self, other: Codex) -> None:
        for rule, paths in other.rules.items():
            repository = self.rules.setdefault(rule, [])
            for path in paths:
                if path not in repository:
                    repository.append(path)

    def export(self) -> list[Path]:
        return [Path(r.head, self.count(p), len(p), r.body) for r, p in self.rules.items()]


def setup_logs() -> None:
    """ Configure `logging` to redirect any log to a file and to the standard output.
    """
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(fmt=logging.Formatter(fmt="%(message)s"))
    stream_handler.setLevel(level=logging.INFO)

    file_handler = logging.FileHandler(
        filename=os.path.join(os.getcwd(), "pyLomma.log"),
        encoding="UTF-8",
        mode="w",
    )
    file_handler.setFormatter(fmt=logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  p%(process)s@%(filename)s:%(lineno)d - %(message)s"))
    file_handler.setLevel(level=logging.DEBUG)

    logger = logging.getLogger()
    logger.addHandler(hdlr=stream_handler)
    logger.addHandler(hdlr=file_handler)
    logger.setLevel(level=logging.DEBUG)


class Strategy:

    @classmethod
    def aggregate(cls, index: Index, paths: list[Path]) -> dict[Triple, list[Path]]:
        result = {}
        for path in paths:
            for ground in path.infer(index):
                result.setdefault(ground.head, []).append(ground)

        return result

    @classmethod
    def apply(cls, aggregation: dict[Triple, list[Path]]):
        return {a: cls.score(p) for a, p in aggregation.items()}

    @staticmethod
    def score(paths: list[Path]) -> float:
        raise NotImplementedError()


class Maximum(Strategy):

    @staticmethod
    def score(paths: list[Path]) -> float:
        return max(p.confidence for p in paths)


class NoisyOR(Strategy):

    @staticmethod
    def score(paths: list[Path]) -> float:
        return 1 - prod((1 - p.confidence) for p in paths)


def parse_args(default: Sequence[str] = None) -> ap.Namespace:
    """Parse arguments from the command line and return them.

    If a `default` sequence of strings is given, the arguments are parsed from there.

    :param default: the optional sequence of strings from where to parse the arguments
    :return: the parsed arguments
    """
    parser = ap.ArgumentParser(description=f"Python implementation of AnyBURL "
                                           f"(https://web.informatik.uni-mannheim.de/AnyBURL/).")
    parser.add_argument("--random-state", "-n", required=False, default=None, type=int,
                        help="The seed to set the random state to")
    parser.add_argument("--input", "-i", required=True, type=str, help="The input .csv file with triples <src,rel,tgt>")
    parser.add_argument("--quality", "-q", required=True, type=float, help="The desired quality on rules confidence")
    parser.add_argument("--saturation", "-s", required=True, type=float, help="The saturation to increase rules length")
    parser.add_argument("--duration", "-d", required=True, type=float, help="The duration in seconds of each stint")
    parser.add_argument("--time", "-t", required=True, type=float, help="The duration in seconds of the experiment")
    parser.add_argument("--workers", "-w", required=False, default=mp.cpu_count(), type=int,
                        help="The number of workers")
    parser.add_argument("--relation", "-r", required=False, default=None, type=str,
                        help="The type of relation to learn")

    parser.add_argument("--force", "-f", required=False, action='store_true', help="Force recomputing results")
    parser.add_argument("--learn", "-l", required=True, type=str,
                        help="File wjere to save the results of the learning.")

    args = parser.parse_args(args=default)
    logging.debug(f"Parsed args: {args}")

    return args


def learn(index: Index, args: ap.Namespace) -> Codex:
    logging.info(f"\n[{0.0:6.2f}%] {0:,} rule/s found")

    num, length, rules = 0, 2, Codex()
    deadline = now() + args.time
    with mp.Pool(processes=args.workers) as pool:
        while True:
            num += 1
            result = pool.aggregate(Codex.discover, args=(num, index, length, args))
            if rules.get_saturation(result) >= args.saturation:
                length += 1

            rules.update(result)
            t = now()
            logging.info(f"[{min(100, 100 - 100 * (deadline - t) / args.time):6.2f}%] {len(rules):,} rule/s found")
            if t >= deadline:
                break

    return rules


if __name__ == '__main__':
    setup_logs()
    args = parse_args([
        "-n", "42",
        "-i", "../data/graph.csv",
        "-q", "0.25",
        "-s", "0.5",
        "-d", "10",
        "-t", "30",
        "-r", "treats",
        # "-w", "2",
        "-l", "rules.pl",
        # "-f",
    ])

    mp.freeze_support()

    # LEARN -- APPLY -- EVAL -- EXPLAIN

    logging.info(f"pyLomma")

    if args.random_state:
        logging.info(f"\nRandom-state set to {args.random_state}")
        seed(args.random_state)

    logging.info(f"\nLoading graph from '{args.input}'...")
    with open(args.input, "r") as file:
        kg = KnowledgeGraph.load(file)
        idx = Index.populate(kg)

    if args.force or not os.path.exists(args.learn):
        rules = learn(idx, args).export()
        with open(args.learn, "w") as file:
            logging.info(f"\nSaving {len(rules)} rule/s to '{args.learn}'...")
            for rule in rules:
                print(rule, file=file)
    else:
        parser = PathParser()
        with open(args.learn, "r") as file:
            logging.info(f"\nLoading rule/s from '{args.learn}'...")
            rules = [parser.parse(line) for line in file]

    aggregations = Maximum.aggregate(idx, rules)
    with open('paths.pl', 'w') as file:
        for paths in aggregations.values():
            for path in paths:
                print(path, file=file)

    predictions = Maximum.apply(aggregations)
    for triple, score in predictions.items():
        print("*", triple, ":", score, "!!! New" if triple not in idx.triples else "")

    logging.info("\nDone.")
