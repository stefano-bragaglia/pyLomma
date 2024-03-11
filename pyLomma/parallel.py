from __future__ import annotations

import csv
import logging
import os
import sys
from argparse import Namespace
from math import prod
from multiprocessing import cpu_count
from multiprocessing import current_process
from multiprocessing import freeze_support
from multiprocessing import Pool
from random import choice
from random import seed
from time import time as now
from typing import Generator
from typing import NamedTuple
from typing import TextIO


class Triple(NamedTuple):
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
    triple: Triple
    inverse: bool

    def get_origin(self) -> str:
        return self.triple.target if self.inverse else self.triple.source

    def get_destination(self) -> str:
        return self.triple.source if self.inverse else self.triple.target

    def replace(self, subst: dict[str, str]) -> Sample:
        return Sample(self.triple.replace(subst), self.inverse)


class Path(NamedTuple):
    expected: bool
    samples: list[Sample]

    def __hash__(self) -> int:
        result = hash(self.expected)
        for atom in self.samples:
            result *= hash(atom)

        return result

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def get_confidence(paths: set[Path]) -> float:
        if not paths:
            return 0.0

        return sum(1 for p in paths if p.expected) / len(paths)

    def _c_subst(self) -> dict[str, str]:
        return {
            self.samples[0].get_origin(): "Y",
            self.samples[0].get_destination(): "X",
        }

    def _ac1x_subst(self) -> dict[str, str]:
        return {
            self.samples[0].get_origin(): self.samples[0].get_origin(),
            self.samples[0].get_destination(): "X",
        }

    def _ac1y_subst(self) -> dict[str, str]:
        return {
            self.samples[0].get_origin(): "Y",
            self.samples[0].get_destination(): self.samples[0].get_destination(),
        }

    def _ac1_subst(self) -> dict[str, str]:
        return {
            self.samples[0].get_origin(): self.samples[0].get_origin(),
            self.samples[0].get_destination(): "X",
            self.samples[-1].get_destination(): self.samples[-1].get_destination(),
        }

    def _ac2_subst(self) -> dict[str, str]:
        return {
            self.samples[0].get_origin(): self.samples[0].get_origin(),
            self.samples[0].get_destination(): "X",
        }

    def is_cyclic(self) -> bool:
        return len(self) > 1 and self.samples[0].get_origin() == self.samples[-1].get_destination()

    def is_valid(self) -> bool:
        terminal = self.samples[0].get_origin()
        visited = [self.samples[0].get_destination()]
        for i, atom in enumerate(self.samples[1:], start=1):
            node = atom.get_destination()
            if node == terminal and i < len(self.samples) - 1 or node in visited:
                return False

            visited.append(node)

        return True

    def generate(self) -> Generator[Path, None, None]:
        if self.is_cyclic():
            yield self.replace(self._c_subst())
            yield self.replace(self._ac1x_subst())
            yield self.replace(self._ac1y_subst())
        else:
            yield self.replace(self._ac1_subst())
            yield self.replace(self._ac2_subst())

    def replace(self, subst: dict[str, str]) -> Path:
        return Path(self.expected, [s.replace(subst) for s in self.samples])


class Rule(NamedTuple):
    head: Triple
    correct: int
    total: int
    body: list[Triple]

    def __repr__(self) -> str:
        if not self.body:
            return f"{self.head} : {self.correct} / {self.total}."

        return f"{self.head} : {self.correct} / {self.total} :- {", ".join(repr(a) for a in self.body)}."

    @property
    def confidence(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @staticmethod
    def score(path: Path, grounds: list[Path]) -> Rule:
        return Rule(
            path.samples[0].triple,
            sum(1 for p in grounds if p.expected),
            len(grounds),
            [a.triple for a in path.samples[1:]]
        )

    def infer(self, index: Index, k: list[Triple] = None, s: dict[str, str] = None) -> Generator[Rule, None, None]:
        k = k or []
        if len(k) == len(self.body):
            yield Rule(self.head.subst(s), self.correct, self.total, k)
        else:
            s = s or {}
            atom = self.body[len(k)].subst(s)
            for triple in index.match(atom):
                if atom.source[0].isupper():
                    s.setdefault(atom.source, triple.source)
                if atom.target[0].isupper():
                    s.setdefault(atom.target, triple.target)
                yield from self.infer(index, [*k, triple], s)


class Index(NamedTuple):
    by_source: dict[str, list[Triple]]
    by_relation: dict[str, list[Triple]]
    by_target: dict[str, list[Triple]]
    by_relation_source: dict[str, dict[str, list[Triple]]]
    by_relation_target: dict[str, dict[str, list[Triple]]]
    triples: list[Triple]

    def __contains__(self, item: Triple | Sample) -> bool:
        if isinstance(item, Sample):
            elem = item.triple
        elif isinstance(item, Triple):
            elem = item
        else:
            raise ValueError("Invalid item")

        return elem in self.by_relation.get(elem.relation, [])

    @staticmethod
    def populate(fp: TextIO) -> Index:
        index = Index({}, {}, {}, {}, {}, [])
        for row in csv.DictReader(fp):
            triple = Triple(**row)
            if triple not in index.triples:
                index.by_source.setdefault(triple.source, []).append(triple)
                index.by_relation.setdefault(triple.relation, []).append(triple)
                index.by_target.setdefault(triple.target, []).append(triple)
                index.by_relation_source.setdefault(triple.relation, {}).setdefault(triple.source, []).append(triple)
                index.by_relation_target.setdefault(triple.relation, {}).setdefault(triple.target, []).append(triple)
                index.triples.append(triple)

        return index

    def get_head(self, relation: str = None) -> Sample:
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

        return Sample(triple, choice([False, True]))

    def get_next(self, path: Path) -> Sample | None:
        if not path.samples:
            raise ValueError("No sample/s found")

        node = path.samples[-1].get_destination()
        inverse = choice([False, True])
        if inverse:
            triples = self.by_target.get(node, [])
        else:
            triples = self.by_source.get(node, [])
        if not triples:
            return None

        return Sample(choice(triples), inverse)

    def sample(self, length: int, relation: str = None) -> Path | None:
        atom = self.get_head(relation)
        path = Path(atom in self, [atom])
        while len(path) < length:
            atom = self.get_next(path)
            if not atom:
                return None

            path.samples.append(atom)
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


class Report(dict[Path, set[Path]]):

    def get_saturation(self, other: Report) -> float:
        if not other:
            return 0.0

        return sum(1 for r in other if r in self) / len(other)

    def merge(self, other: Report) -> None:
        for rule, paths in other.items():
            self.setdefault(rule, set()).update(paths)

    def filter(self, quality: float) -> Report:
        return Report({r: p for r, p in self.items() if r.get_confidence(p) >= quality})


class Policy:

    @classmethod
    def aggregate(cls, index: Index, rules: list[Rule]) -> dict[Triple, list[Rule]]:
        result = {}
        for rule in rules:
            for ground in rule.infer(index):
                result.setdefault(ground.head, []).append(ground)

        return result

    @classmethod
    def apply(cls, aggregation: dict[Triple, list[Rule]]):
        return {a: cls.score(p) for a, p in aggregation.items()}

    @staticmethod
    def score(grounds: list[Rule]) -> float:
        raise NotImplementedError()


class Maximum(Policy):

    @staticmethod
    def score(grounds: list[Rule]) -> float:
        return max(p.confidence for p in grounds)


class NoisyOR(Policy):

    @staticmethod
    def score(grounds: list[Rule]) -> float:
        return 1 - prod((1 - p.confidence) for p in grounds)


def is_constant(value: str) -> bool:
    return value[0].islower()


def learn(index: Index, args: Namespace) -> list[Rule]:
    logging.info(f"\nLearning rules (length: "
                 f"min={args.min_length}, max-ac={args.max_acyclic_length}, max-c={args.max_cyclic_length})...")

    length = 2  # args.min_length
    with Pool(args.workers) as pool:
        logging.info(f"\n[{0.0:6.2f}%] {0:,} rule/s found")
        deadline = now() + args.duration
        while True:
            args.saturate = False
            results = [
                pool.apply_async(_discover, args=(index, length, args), callback=_update)
                for _ in range(args.workers)
            ]
            for r in results:
                r.wait()
            if args.saturate:
                length = length + 1
                logging.info(f"\nResult set saturated: extending sampling length to '{length}'...")

            time = now()
            logging.info(
                f"[{min(100, 100 - 100 * (deadline - time) / args.duration):6.2f}%] {len(args.rules):,} rule/s found")
            if time >= deadline:
                break

    rules = [Rule.score(r, p) for r, p in args.rules.items()]
    with open("rules.pl", "w") as file:
        for score in sorted(rules, key=lambda x: (-x.confidence, repr(x.head), repr(x.body))):
            print(score, file=file)

    return rules


def _discover(index: Index, length: int, args: Namespace) -> dict[Path, set[Path]]:
    setup_logs()
    seed(args.random_state)
    logging.debug(f"\nSampling session '{args.number}.{current_process().pid}'...")

    report = Report()
    if length < args.max_acyclic_length or length < args.max_cyclic_length:
        deadline = now() + args.span
        while True:
            path = index.sample(length, args.relation)
            if path:
                for rule in path.generate():
                    max_limit = args.max_cyclic_length if rule.is_cyclic() else args.max_acyclic_length
                    if len(rule) < max_limit:
                        report.setdefault(rule, set()).add(path)

            if now() >= deadline:
                break

    return report


def _update(report: Report) -> None:
    logging.debug(f"\nCallback session '{args.number}.{current_process().pid}'...")

    args.number += 1
    with open(f"simple_rules_{args.number}.pl", "w") as file:
        for rule, paths in report.items():
            print(rule, file=file)
            for path in paths:
                print("%", path, file=file)
            print(file=file)

    current = report.filter(args.quality)
    if current.get_saturation(args.rules) >= args.saturation:
        args.saturate = True
    args.rules.merge(current)


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


if __name__ == '__main__':
    setup_logs()

    args = Namespace(
        apply='ranking.csv',
        duration=30,
        input='../data/graph.csv',
        max_acyclic_length=4,
        max_cyclic_length=11,
        min_length=3,
        number=0,
        policy='maximum',
        # policy='noisy-or',
        quality=0.25,
        random_state=42,
        relation='treats',
        rules=Report(),
        saturate=False,
        saturation=0.5,
        span=3,
        workers=cpu_count(),
    )

    freeze_support()

    logging.info(f"pyLomma")

    if args.random_state:
        logging.info(f"\nRandom-state set to {args.random_state}")
        seed(args.random_state)

    logging.info(f"\nLoading graph from '{args.input}'...")
    with open(args.input, "r") as file:
        index = Index.populate(file)

    rules = learn(index, args)

    aggregations = Policy.aggregate(index, rules)
    match args.policy:
        case "maximum":
            logging.info("\nApplying scores using 'maximum' policy...")
            predictions = Maximum.apply(aggregations)
        case "noisy-or":
            logging.info("\nApplying scores using 'noisy-or' policy...")
            predictions = NoisyOR.apply(aggregations)
        case _:
            raise ValueError(f"Unexpected policy: {args.policy}")

    with open(args.apply, 'w') as file:
        logging.info(f"\nSaving ranking to '{args.apply}'...")
        writer = csv.DictWriter(file, fieldnames=["target", "score", "new"])
        writer.writeheader()
        for target, score in predictions.items():
            writer.writerow({
                "target": target,
                "score": score,
                "new": target not in index,
            })

    logging.info("\nDone.")
