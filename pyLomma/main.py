from __future__ import annotations

import argparse as ap
import csv
import logging
import multiprocessing as mp
import os
import sys
from random import choice
from random import seed
from time import time as now
from typing import Generator
from typing import NamedTuple
from typing import Sequence
from typing import TextIO


def is_constant(value: str) -> bool:
    return value[0].islower()


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

    def get_subst(self, const: Triple) -> dict[str, str]:
        result = {}
        if self.source[0].isupper():
            result.setdefault(self.source, const.source)
        if self.target[0].isupper():
            result.setdefault(self.target, const.target)

        return result


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
    """ A path is a sequence of samples.
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
            yield Rule.generalize(self, self._c_subst())
            yield Rule.generalize(self, self._ac1x_subst())
            yield Rule.generalize(self, self._ac1y_subst())
        else:
            yield Rule.generalize(self, self._ac1_subst())
            yield Rule.generalize(self, self._ac2_subst())

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

    head: Sample
    body: list[Sample]

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
    def generalize(path: Path, subst: dict[str, str] = None) -> Rule:
        return Rule(path.head.replace(subst), [a.replace(subst) for a in path.body])


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

    def next_step(self, path: Path) -> Sample | None:
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

    def random_walk(self, length: int, relation: str = None) -> Path | None:
        """ Return a path by performing a random walk from an example.

        :param length: the desired length for the path (min 2)
        :param relation: the optional relation to learn (if missing, learn any relation)
        :return: a sampled path or None
        """
        head, expected = self.sample(relation)
        path = Path(head, [], expected)
        while len(path) < length:
            atom = self.next_step(path)
            if not atom:
                return None

            path.body.append(atom)
            if not path.is_valid():
                return None

        return path

    def find(self, atom: Triple) -> Generator[Triple, None, None]:
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

    def query(self, rule: Rule, subst: dict[str, str] = None, pos: int = None) -> Generator[Triple, None, None]:
        pos = pos or 0
        if pos >= len(rule):
            yield Triple(
                subst.get(rule.head.triple.source, rule.head.triple.source),
                rule.head.triple.relation,
                subst.get(rule.head.triple.target, rule.head.triple.target),
            )
        else:
            subst = subst or {}
            for triple in self.find(rule.body[pos].triple):
                if pos <= 0 or rule.body[pos - 1].get_destination() == rule.body[pos].get_origin():
                    subst.update(rule.body[pos].triple.get_subst(triple))
                    yield from self.query(rule, subst, pos + 1)


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

    args = parser.parse_args(args=default)
    logging.debug(f"Parsed args: {args}")

    return args


def main(args: ap.Namespace):
    logging.info(f"pyLomma")

    if args.random_state:
        logging.info(f"\nRandom-state set to {args.random_state}")
        seed(args.random_state)

    logging.info(f"\nLoading '{args.input}'...")
    with open(args.input, "r") as file:
        kg = KnowledgeGraph.load(file)
        idx = Index.populate(kg)

    num, length, rules = 0, 2, {}
    deadline = now() + args.time
    with mp.Pool(processes=args.workers) as pool:
        while True:
            result = pool.apply(session, args=(idx, length, args))
            if result and sum(1 for r in result if r in rules) / len(result) > args.saturation:
                length += 1

            num += 1
            with open(f"rules_{num}.pl", "w") as file:
                for rule, paths in result.items():
                    file.write(f"{rule.head}"
                               f" : {sum(1 for p in paths if p.expected)}"
                               f" / {len(paths)}"
                               f" :- {", ".join(repr(a) for a in rule.body)}.\n")

            for rule, paths in result.items():
                rules.setdefault(rule, set()).update(paths)

            current = now()
            logging.info(
                f"{min(100, 100 - 100 * (deadline - current) / args.time):6.2f}% - {len(rules):,} rule/s found")

            if current >= deadline:
                break

        rule = list(rules.keys())[0]
        print(rule)
        print(sorted(set(idx.query(rule))))

    return rules


def session(index: Index, length: int, args: ap.Namespace) -> dict[Rule, set[Path]]:
    table = {}
    deadline = now() + args.duration
    while now() <= deadline:
        path = index.random_walk(length, args.relation)
        if path:
            for rule in path.generalize():
                table.setdefault(rule, set()).add(path)

    result = {
        r: s
        for r, s in table.items()
        if sum(1 for p in s if p.expected) / len(s) >= args.quality
    }

    return result


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
    ])

    mp.freeze_support()

    res = main(args)

    ### Nice to have a place where to store rules, the triples that they can predict, the triples that were sampled
    # notice that all these triples can be used to compute the confidence on the rule
    ### the rules can be sorted by confidence
    ### for each predictable triple, consider the confidence of all the rules that produce it
    ### use strategies to combine them: maximum, noisyor
    ### for each triple, get all the paths the rules that produce it can produce, combine them in a graph

    with open(f"rules.pl", "w") as file:
        for rule, paths in sorted(res.items(),
                                  key=lambda x: (-sum(1 for p in x[1] if p.expected) / len(x[1]), repr(x[0]))):
            print(f"{rule.head}"
                  f" : {sum(1 for p in paths if p.expected)}"
                  f" / {len(paths)}"
                  f" :- {", ".join(repr(a) for a in rule.body)}.", file=file)
            # for path in sorted(paths, key=lambda x: repr(x)):
            #     print(f"% {path.head.triple} :"
            #           f" {path.expected} :-"
            #           f" {", ".join(repr(a.triple) for a in path.body)}.", file=file)
            # print(file=file)

    logging.info("\nDone.")
