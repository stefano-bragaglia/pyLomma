from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime
from random import choice
from typing import Generator
from typing import NamedTuple
from typing import TextIO

from __init__ import __version__


class Triple(NamedTuple):
    """ A (semantic) triple contains a subject or 'source', a predicate or 'relation', and an object or 'destination'.
    """

    source: str
    relation: str
    target: str

    def __repr__(self) -> str:
        return f"{self.relation}({self.source},{self.target})"


class Sample(NamedTuple):
    """ A sample is a triple randomly selected during a random walk.

    Inverse is a flag indicating whether the triple has been traversed according
    its natural direction or not.
    """

    triple: Triple
    inverse: bool

    @staticmethod
    def create(candidates: list[Triple], inverse: bool = None) -> Sample:
        """ Create a sample from a list of candidates.

        If no flag 'inverse' is given, it is randomly determined.

        :param candidates: a list of triples to randomly choose a candidate from
        :param inverse: a flag indicating the direction to traverse the candidate
        :return: a sample
        """
        candidate = choice(candidates)
        if inverse is None:
            inverse = choice([False, True])

        return Sample(candidate, inverse)

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


class Rule(NamedTuple):
    """ A rule is a sequence of samples with at least 1 variable.

    A lowercase string represent a constant; an uppercase string represent a variable.
    """

    head: Sample
    body: list[Sample]


class Path(NamedTuple):
    """ A path is a sequence of samples.
    """

    head: Sample
    body: list[Sample]

    def __len__(self) -> int:
        return len(self.body)

    def __repr__(self) -> str:
        body = ", ".join(repr(a.triple) for a in self.body)
        if not body:
            return f"{self.head.triple}."

        return f"{self.head.triple} :- {body}."

    @staticmethod
    def create(examples: list[Triple]) -> Path:
        """ Start a path from a list of examples and return it.

        :param examples: a list of examples
        :return: a path
        """
        return Path(Sample.create(examples), [])

    def append(self, atom: Sample) -> None:
        """ Append a sample to the body of the path.

        :param atom: the sample to append to the body of the path
        """
        self.body.append(atom)

    def generalize(self) -> Generator[Rule, None, None]:
        pass

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

    def next_step(self, index: Index) -> Sample | None:
        """ Return a possible next step in a random walk for this path.

        :param index: the index containing the potential candidates for the next step
        :return: the next step or None if no next step is possible
        """
        if not self.body:
            node = self.head.get_destination()
        else:
            node = self.body[-1].get_destination()

        inverse = choice([False, True])
        if inverse:
            triples = index.triples_by_target.get(node, [])
        else:
            triples = index.triples_by_source.get(node, [])

        if not triples:
            return None

        return Sample.create(triples, inverse)


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
        start = datetime.now()
        reader = csv.DictReader(fp)
        assert reader.fieldnames == ["source", "relation", "target"], \
            "Expected a .csv file with fields: 'source', 'relation', and 'target'"

        facts = [Triple(**t) for t in reader]
        result = KnowledgeGraph(facts)
        logging.info(f"  {len(facts):,} triple/s loaded in {datetime.now() - start}s")

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
        start = datetime.now()
        facts = [t._asdict() for t in sorted(set(self.facts), key=lambda x: (x.relation, x.source, x.target))]
        writer = csv.DictWriter(fp, fieldnames=["source", "relation", "target"])
        writer.writeheader()
        writer.writerows(facts)
        logging.info(f"  {len(facts):,} triple/s saved in {datetime.now() - start}s")


class Index(NamedTuple):
    """ An index is a collection of triples indexed by source, relation, and target.
    """

    triples: list[Triple]
    triples_by_source: dict[str, list[Triple]]
    triples_by_relation: dict[str, list[Triple]]
    triples_by_target: dict[str, list[Triple]]

    @staticmethod
    def populate(kg: KnowledgeGraph) -> Index:
        """ Populates an index from a knowledge graph.

        :param kg: the knowledge graph to index
        :return: the index on the knowledge graph
        """
        start = datetime.now()
        triples, triples_by_source, triples_by_relation, triples_by_target = [], {}, {}, {}
        for triple in kg.facts:
            if triple not in triples:
                triples_by_source.setdefault(triple.source, []).append(triple)
                triples_by_relation.setdefault(triple.relation, []).append(triple)
                triples_by_target.setdefault(triple.target, []).append(triple)
                triples.append(triple)

        result = Index(triples, triples_by_source, triples_by_relation, triples_by_target)
        logging.info(f"  {len(triples):,} unique triple/s ("
                     f"{len(triples_by_source):,} source/s, "
                     f"{len(triples_by_relation):,} relation/s, "
                     f"{len(triples_by_target):,} target/s"
                     f") indexed in {datetime.now() - start}s")

        return result

    def sample(self, examples: list[Triple], length: int) -> Path | None:
        """ Return a path of given length by sampling an example and performing a random walk.

        :param examples: the examples to sample
        :param length: the desired length for the path (min 2)
        :return: a sampled path or None
        """
        path = Path.create(examples)
        while len(path) < length:
            atom = path.next_step(self)
            if not atom:
                return None

            path.append(atom)
            if not path.is_valid():
                return None

        return path


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

    logging.info(f"pyLomma {__version__}")

    filename = "../data/graph.csv"
    logging.info(f"\nLoading '{filename}'...")
    with open(filename, "r") as file:
        graph = KnowledgeGraph.load(file)

    index = Index.populate(graph)
    examples = index.triples_by_relation.get("treats", index.triples)
    length = 2
    path = index.sample(examples, length)

    print(path)
    print("Done.")
