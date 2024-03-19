from __future__ import annotations

import gzip
import hashlib
import json
import os
from urllib.parse import urlparse

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from pyLomma.utils import benchmark

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Hetionet:
    RESOURCES = {
        "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0-metagraph.json":
            "29de89d574d601161dc4ddcdb2975a66",
        "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz":
            "b4b2ab8890d403c2077a65e0f34fc4d9",
        "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv":
            "d2789861ab5b977dcee3c23ac9c9f7ce",
    }

    @staticmethod
    def md5_hex(filename: str) -> str:
        with open(filename, "rb") as f:
            return hashlib.file_digest(f, "md5").hexdigest()

    @staticmethod
    def download(folder: str) -> None:
        with benchmark("download hetionet"):
            for url, digest in Hetionet.RESOURCES.items():
                _, filename = os.path.split(urlparse(url).path)
                fullname = os.path.join(folder, filename)
                if not os.path.exists(fullname) or Hetionet.md5_hex(fullname) != digest:
                    with open(fullname, "wb") as f:
                        with requests.get(url) as r:
                            f.write(r.content)
                    print(f"* Got '{filename}' ({Hetionet.md5_hex(fullname)})")

    @staticmethod
    def _get_table(folder: str) -> dict[str, str]:
        url = list(Hetionet.RESOURCES.keys())[0]
        _, filename = os.path.split(urlparse(url).path)
        fullname = os.path.join(folder, filename)
        with open(fullname, "r") as f:
            dataset = json.load(f)
            data = dataset['kind_to_abbrev']

        return {
            f"{data[src]}{data[rel]}{'' if dir == 'both' else '>'}{data[tgt]}": rel
            for src, tgt, rel, dir in dataset["metaedge_tuples"]
        }

    @staticmethod
    def _get_edges(folder: str) -> pd.DataFrame:
        table = Hetionet._get_table(folder)
        url = list(Hetionet.RESOURCES.keys())[1]
        _, filename = os.path.split(urlparse(url).path)
        fullname = os.path.join(folder, filename)
        with gzip.open(fullname, "rb") as f:
            content = f.read()

        srcs, rels, tgts = [], [], []
        text = content.decode()
        for i, line in enumerate(text.split("\r\n")):
            if i > 0 and line:
                src, rel, tgt = line.split("\t")
                srcs.append(src)
                rels.append(table[rel])
                tgts.append(tgt)

        return pd.DataFrame({"src": srcs, "rel": rels, "tgt": tgts})

    @staticmethod
    def _get_nodes(folder: str) -> pd.DataFrame:
        url = list(Hetionet.RESOURCES.keys())[2]
        _, filename = os.path.split(urlparse(url).path)
        fullname = os.path.join(folder, filename)

        return pd.read_csv(fullname, sep="\t")

    @staticmethod
    def populate(folder: str) -> Graph:
        with benchmark("load hetionet"):
            return Graph(
                Hetionet._get_edges(folder),
                Hetionet._get_nodes(folder),
            )


class Graph:

    @staticmethod
    def populate(filename: str) -> Graph:
        abbrev = {
            "Biological Process": "BP",
            "Cellular Component": "CC",
            "causes": "c",
            "Pharmacologic Class": "PC",
            "Molecular Function": "MF",
            "palliates": "p",
            "downregulates": "d",
            "expresses": "e",
            "Gene": "G",
            "covaries": "c",
            "upregulates": "u",
            "presents": "p",
            "Anatomy": "A",
            "Symptom": "S",
            "Pathway": "PW",
            "treats": "t",
            "localizes": "l",
            "Disease": "D",
            "participates": "p",
            "binds": "b",
            "includes": "i",
            "associates": "a",
            "Compound": "C",
            "interacts": "i",
            "resembles": "r",
            "regulates": "r",
            "Side Effect": "SE"
        }
        abbrev = {k: v for v, k in abbrev.items() if k.islower()}

        edges = pd.read_csv(filename, delimiter="\t").rename(columns={
            "source": "src", "metaedge": "rel", "target": "tgt"})
        edges["rel"] = [abbrev[''.join(c for c in r if c.islower())] for r in edges["rel"]]

        return Graph(edges)

    def __init__(self, edges: pd.DataFrame, nodes: pd.DataFrame = None):
        self.edges = edges
        assert list(self.edges.columns) == ["src", "rel", "tgt"]

        if nodes is None:
            self.nodes = pd.DataFrame({"id": [], "name": [], "kind": []})
        else:
            self.nodes = nodes
        assert list(self.nodes.columns) == ["id", "name", "kind"]

    def get_examples(self, rel: str) -> pd.DataFrame:
        with benchmark("extract examples"):
            pos = self.edges.loc[self.edges.rel == rel]
            srcs = pos[['src']].drop_duplicates()
            tgts = pos[['tgt']].drop_duplicates()
            examples = srcs.assign(rel=rel).merge(tgts, how="cross")
            result = examples.merge(pos, how="left", indicator='exp')
            result['exp'] = result['exp'] == 'both'

            return result

    def sample_canon(self, examples: pd.DataFrame, length: int) -> list[pd.DataFrame]:
        with benchmark("prepare head"):
            head = examples.sample()
            head = pd.concat([
                head.assign(orig=lambda x: x.src, next=lambda x: x.tgt),
                head.assign(orig=lambda x: x.tgt, next=lambda x: x.src),
            ], ignore_index=True).sample()

        path = [head]
        for i in range(1, length):
            with benchmark(f"prepare body_{i}"):
                connected_fwd = self.edges.src.isin(path[-1].next)
                connected_bwd = self.edges.tgt.isin(path[-1].next)
                different_fwd = ~self.edges.tgt.isin(head.next)
                different_bwd = ~self.edges.src.isin(head.next)
                for step in path[1:]:
                    different_fwd = different_fwd & ~self.edges.tgt.isin(step.next)
                    different_bwd = different_bwd & ~self.edges.src.isin(step.next)
                    if i < length - 1:  # not last --> non-cyclic (last: whatever)
                        different_fwd = different_fwd & ~self.edges.tgt.isin(head.orig)
                        different_bwd = different_bwd & ~self.edges.src.isin(head.orig)

                path.append(pd.concat([
                    self.edges.loc[connected_fwd & different_fwd].assign(next=lambda x: x.tgt),
                    self.edges.loc[connected_bwd & different_bwd].assign(next=lambda x: x.src),
                ], ignore_index=True).sample())

        return path

    def sample(self, examples: pd.DataFrame, length: int) -> list[pd.DataFrame]:
        with benchmark("prepare edges"):
            edges = self.edges.loc[self.edges.src != self.edges.tgt]
            atoms = pd.concat([
                edges.assign(prev=lambda x: x.src, next=lambda x: x.tgt),
                edges.assign(prev=lambda x: x.tgt, next=lambda x: x.src),
            ], ignore_index=True).drop_duplicates()

        with benchmark("prepare head"):
            examples = examples.loc[examples.src != examples.tgt]
            head = pd.concat([
                examples.assign(orig=lambda x: x.src, next=lambda x: x.tgt),
                examples.assign(orig=lambda x: x.tgt, next=lambda x: x.src),
            ], ignore_index=True).drop_duplicates()
            paths = [head]

        for i in range(1, length):
            with benchmark(f"prepare body_{i}"):
                connected = atoms.prev.isin(paths[-1].next)
                different = ~atoms.next.isin(head.next)
                for step in paths[1:]:
                    different = different & ~atoms.next.isin(step.next)
                    if i < length - 1:  # not last --> non-cyclic (last: whatever)
                        different = different & ~atoms.next.isin(head.orig)
                paths.append(atoms.loc[connected & different])

        return paths


if __name__ == '__main__':
    Hetionet.download("../data")
    graph = Hetionet.populate("../data")
    examples = graph.get_examples("treats")
    train, test = train_test_split(examples, test_size=999 / 1000, random_state=42, shuffle=True, stratify=examples.exp)
    with benchmark("prepare canon sampling"):
        something = graph.sample_canon(train, 11)
    with benchmark("prepare sampling"):
        something = graph.sample(train, 11)
        # for anything in something:
        #     print(anything)

    print("Done.")
