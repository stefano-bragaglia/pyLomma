from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from random import choice
from typing import Generator
from typing import NamedTuple


def setup_logs() -> None:
    """ Configure `logging` to redirect any log to a file and to the standard output.
    """
    file_handler = logging.FileHandler(filename="pyLomma.log", mode="w", encoding="UTF-8")
    file_handler.setFormatter(fmt=logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  p%(process)s@%(filename)s:%(lineno)d - %(message)s"))
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(fmt=logging.Formatter(fmt="%(message)s"))
    stream_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


class Triple(NamedTuple):
    """ A unit of information in a Knowledge Graph, consisting of 3 values: source, relation, and target.
    """

    source: str
    relation: str
    target: str

    def __repr__(self) -> str:
        return f"{self.relation}({self.source},{self.target})"


class Sample(NamedTuple):

    triple: Triple
    inverse: bool

    def __repr__(self) -> str:
        if self.inverse:
            return f"~{self.triple.relation}({self.triple.target},{self.triple.source})"

        return f"{self.triple.relation}({self.triple.source},{self.triple.target})"


class Path(NamedTuple):

    samples: list[Sample]
    correct: bool


class KnowledgeGraph:
    """ A collection of triples to describe a problem.
    """

    @staticmethod
    def load(filename: str) -> KnowledgeGraph:
        """ Load the triples in the .csv file with given `filename` into a KnowledgeGraph.

        :param filename: the name of the .csv file containing the triples to load
        :return: the KnowledgeGraph containing the triples loaded from file
        """
        logging.info(f"Loading triples from '{filename}'...")

        start = datetime.now()
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == ["source", "relation", "target"], \
                "Expected a .csv file with fields: 'source', 'relation', and 'target'"

            triples = [Triple(**r) for r in reader]
            result = KnowledgeGraph(triples)
            logging.info(f"  {len(triples):,} triple/s loaded in {datetime.now() - start}s\n")

        return result

    def __init__(self, triples: list[Triple] = None):
        self.triples = triples or []

    def dump(self, filename: str) -> None:
        """ Save the triples in this KnowledgeGraph into a .csv file with given `filename`.

        :param filename: the name of the .csv file to store the triples in this KnowledgeGraph
        """
        logging.info(f"Saving triples to '{filename}'...")

        start = datetime.now()
        triples = [t._asdict() for t in sorted(self.triples, key=lambda x: (x.relation, x.source, x.target))]
        with open(filename, "w") as file:
            writer = csv.DictWriter(file, fieldnames=["source", "relation", "target"])
            writer.writeheader()
            writer.writerows(triples)
            logging.info(f"  {len(self.triples):,} triple/s saved in {datetime.now() - start}s\n")


class Index:
    """ Index to quickly access nodes and edges in a graph.
    """

    @staticmethod
    def populate(graph: KnowledgeGraph) -> Index:
        logging.info(f"Populating index...")

        start = datetime.now()
        triples = set()
        triples_by_source, triples_by_target = {}, {}
        sources_by_relation, targets_by_relation = {}, {}
        for triple in set(graph.triples):
            triples.add(triple)
            triples_by_source.setdefault(triple.source, set()).add(triple)
            triples_by_target.setdefault(triple.target, set()).add(triple)
            sources_by_relation.setdefault(triple.relation, set()).add(triple.source)
            targets_by_relation.setdefault(triple.relation, set()).add(triple.target)

        result = Index(list(triples), triples_by_source, triples_by_target, sources_by_relation, targets_by_relation)
        logging.info(f"  {len(triples_by_source):,} source node/s,"
                     f" {len(sources_by_relation):,} relation/s, and"
                     f" {len(triples_by_target):,} target node/s"
                     f" from {len(triples):,} triple/s indexed in {datetime.now() - start}s\n")

        return result

    def __init__(
            self,
            triples: list[Triple] = None,
            triples_by_source: dict[str, list[Triple]] = None,
            triples_by_target: dict[str, list[Triple]] = None,
            sources_by_relation: dict[str, list[str]] = None,
            targets_by_relation: dict[str, list[str]] = None,
    ):
        self.triples = triples or [],
        self.triples_by_source = triples_by_source or {}
        self.triples_by_target = triples_by_target or {}
        self.sources_by_relation = sources_by_relation or {}
        self.targets_by_relation = targets_by_relation or {}

    def sample(self, length: int) -> Generator[Path, None, None]:

        # Not like this: they are all true in this way
        inverse = choice([False, True])
        triple = choice(self.triples)
        correct = triple in self.triples

        samples = [Sample(triple, inverse)]
        for i in range(length):
            node = samples[i].triple.source if samples[i].inverse else samples[i].triple.target

            inverse = choice([False, True])
            triple = choice(self.triples_by_source[node]) if inverse else choice(self.triples_by_target[node])
            samples.append(Sample(triple, inverse))

        yield Path(samples, correct)


if __name__ == '__main__':
    setup_logs()
    graph = KnowledgeGraph.load("example.csv")
    graph.dump("example.csv")
    Index.populate(graph)
