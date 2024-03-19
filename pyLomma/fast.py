from __future__ import annotations

import json
import random
from random import choice
from typing import Generator

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import as_filename
from utils import benchmark
from utils import download


class Graph:

    @staticmethod
    def hetionet(folder: str) -> Graph:
        url = 'https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0-metagraph.json'
        download(folder, url, '29de89d574d601161dc4ddcdb2975a66')
        with open(as_filename(folder, url), 'r') as f:
            dataset = json.load(f)
            abb, tuples = dataset['kind_to_abbrev'], dataset['metaedge_tuples']
            convert = {f'{abb[s]}{abb[r]}{'' if d == 'both' else '>'}{abb[t]}': r for s, t, r, d in tuples}

        url = 'https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz'
        download(folder, url, 'b4b2ab8890d403c2077a65e0f34fc4d9')
        edges = pd.read_csv(as_filename(folder, url), sep='\t').drop_duplicates()
        edges.insert(2, 'kind', edges.pop('metaedge').apply(lambda x: convert[x]))

        url = 'https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv'
        download(folder, url, 'd2789861ab5b977dcee3c23ac9c9f7ce')
        nodes = pd.read_csv(as_filename(folder, url), sep='\t').drop_duplicates()
        nodes = nodes.rename(columns={'id': 'name', 'name': 'label'})

        return Graph(edges, nodes)

    def __init__(self, edges: pd.DataFrame, nodes: pd.DataFrame):
        assert list(edges.columns) == ['source', 'target', 'kind']
        assert list(nodes.columns) == ['name', 'label', 'kind']
        with benchmark('graph loaded'):
            self.graph = ig.Graph.DataFrame(edges, directed=False, vertices=nodes, use_vids=False)

    def get_examples(self, rel: str) -> pd.DataFrame:
        with benchmark('extract examples'):
            pos = pd.DataFrame([{
                'source': self.graph.vs[e.source]['name'],
                'target': self.graph.vs[e.target]['name'],
                'kind': rel
            } for e in self.graph.es.select(kind=rel)])

            srcs = pos[['source']].drop_duplicates()
            tgts = pos[['target']].drop_duplicates()
            result = srcs.merge(tgts.assign(kind=rel), how='cross')
            result = result.merge(pos, how='left', indicator='expected')
            result = result.loc[result.source != result.target].drop_duplicates()
            result.insert(0, 'expected', result.pop('expected').apply(lambda x: x == 'both'))

            return result

    def sample(self, examples: pd.DataFrame, length: int) -> tuple[bool, list[tuple[str, str, str]]] | None:
        assert list(examples.columns) == ['expected', 'source', 'target', 'kind']

        try:
            expected, source, target, kind = examples.sample().values[0]

            start = choice([source, target])

            origin = self.graph.vs.find(target if start == source else source)
            previous = [self.graph.vs.find(start)]

            path = [(source, target, kind)]
            for i in range(1, length):
                neighbors = self.graph.neighbors(previous[-1])
                index = choice(neighbors)
                following = self.graph.vs[index]
                if any(following == p for p in previous) or (i < length - 1 and following == origin):
                    return None

                relation = choice(self.graph.es.select(_source=previous, _target=following))
                path.append((previous[-1]['name'], following['name'], relation['kind']))
                previous.append(following)
        except ValueError | IndexError:
            return None

        return expected, path

    @staticmethod
    def generalize(path: list[tuple[str, str, str]]) -> Generator[list[tuple[str, str, str]], None, None]:
        origin = path[0][1] if path[0][0] in path[1] else path[0][0]
        destination = path[-1][1] if path[-1][0] in path[-2] else path[-1][0]
        if origin == destination:  # cyclic
            pass
        else:
            pass




if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    g = Graph.hetionet('../data')
    ex = g.get_examples('treats')
    train, test = train_test_split(ex, test_size=1 / 3, random_state=42, shuffle=True, stratify=ex.expected)

    while True:
        found = g.sample(train, 3)
        if found:
            expected, path = found
            for rule in g.generalize(path):
                print(rule)

        break

    print('Done.')
