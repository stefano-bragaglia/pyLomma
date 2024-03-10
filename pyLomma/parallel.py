from __future__ import annotations

import csv
from argparse import Namespace
from multiprocessing import cpu_count
from multiprocessing import freeze_support
from multiprocessing import Pool
from random import choice
from random import seed
from time import time as now
from typing import Generator
from typing import NamedTuple


class Triple(NamedTuple):
    source: str
    relation: str
    target: str

    @staticmethod
    def convert(value: str, subst: dict[str, str]) -> str:
        return subst.setdefault(value, f"A{2 + sum(1 for x in subst.values() if x.startswith("A"))}")

    def replace(self, subst: dict[str, str]) -> Triple:
        return Triple(self.convert(self.source, subst), self.relation, self.convert(self.target, subst))


class Sample(NamedTuple):
    triple: Triple
    inverse: bool

    def get_origin(self) -> str:
        return self.triple.target if self.inverse else self.triple.source

    def get_destination(self) -> str:
        return self.triple.source if self.inverse else self.triple.target


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

    def generate(self) -> Generator[Rule, None, None]:
        if self.is_cyclic():
            yield self.generalize(self._c_subst())
            yield self.generalize(self._ac1x_subst())
            yield self.generalize(self._ac1y_subst())
        else:
            yield self.generalize(self._ac1_subst())
            yield self.generalize(self._ac2_subst())

    def generalize(self, subst: dict[str, str]) -> Rule:
        triples = [a.triple.replace(subst) for a in self.samples]

        return Rule(triples[0], triples[1:])


class Rule(NamedTuple):
    head: Triple
    body: list[Triple]

    def __hash__(self) -> int:
        result = hash(self.head)
        for atom in self.body:
            result *= hash(atom)

        return result

    @staticmethod
    def get_confidence(paths: list[Path]) -> float:
        if not paths:
            return 0.0

        return sum(1 for p in paths if p.expected) / len(paths)


class Scoring(NamedTuple):
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


def _discover(index, length, relation) -> dict[Rule, set[Path]]:
    print(f'myfunc is called with {index}, {length}')
    seed(42)

    codex = Codex()
    deadline = now() + 3
    while True:
        path = index.sample(length, relation)
        if path:
            for rule in path.generate():
                codex.setdefault(rule, set()).add(path)

        if now() >= deadline:
            break

    return codex


class Codex(dict[Rule, set[Path]]):

    def get_saturation(self, other: Codex) -> float:
        if not other:
            return 0.0

        return sum(1 for r in other if r in self) / len(other)

    def merge(self, other: Codex) -> None:
        for rule, paths in other.items():
            self.setdefault(rule, set()).update(paths)


def _update(x):
    print(f'mycallback is called with {x}')

    status.number += 1
    with open(f"simple_rules_{status.number}.pl", "w") as file:
        for rule, paths in x.items():
            print(rule, file=file)
            for path in paths:
                print("%", path, file=file)
            print(file=file)

    current = Codex({r: p for r, p in x.items() if r.get_confidence(p) >= status.quality})
    if current.get_saturation(status.rules) >= status.saturation:
        status.saturate = True
    status.rules.merge(current)


def learn(index, args):
    length = 2
    with Pool(args.workers) as pool:
        deadline = now() + 10
        while True:
            print(length)
            args.saturate = False
            results = [
                pool.apply_async(_discover, args=(index, length, args.relation), callback=_update)
                for _ in range(args.workers)
            ]
            for r in results:
                r.wait()
            if args.saturate:
                print('saturation!!!')
                length = length + 1

            if now() >= deadline:
                break

    for rule, paths in args.rules.items():
        print("*", rule)
        for path in paths:
            print(" ", "|", path)
        print()


if __name__ == '__main__':
    seed(42)
    freeze_support()

    status = Namespace(
        number=0,
        quality=0.25,
        relation='treats',
        rules=Codex(),
        saturate=False,
        saturation=0.5,
        workers=cpu_count(),
    )

    with open("../data/graph.csv", "r") as file:
        index = Index({}, {}, {}, {}, {}, [])
        for row in csv.DictReader(file):
            triple = Triple(**row)
            if triple not in index.triples:
                index.by_source.setdefault(triple.source, []).append(triple)
                index.by_relation.setdefault(triple.relation, []).append(triple)
                index.by_target.setdefault(triple.target, []).append(triple)
                index.by_relation_source.setdefault(triple.relation, {}).setdefault(triple.source, []).append(triple)
                index.by_relation_target.setdefault(triple.relation, {}).setdefault(triple.target, []).append(triple)
                index.triples.append(triple)

    learn(index, status)
