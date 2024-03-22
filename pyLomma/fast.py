from __future__ import annotations

import argparse as ap
import gzip as gz
import hashlib as hl
import io
import json
import logging as log
import multiprocessing as mp
import os
import random as rnd
import sys
from collections import Counter
from math import floor
from math import log1p
from math import prod
from timeit import default_timer as now
from typing import Generator
from typing import Sequence

import igraph as ig
import numpy as np
import pandas as pd
import requests as req
from sklearn.model_selection import train_test_split

Triple = tuple[str, str, str]
Walk = tuple[Triple, ...]
Report = dict[Walk, dict[Walk, bool]]
Aggregation = dict[Triple, dict[Walk, float]]
Ranking = dict[Triple, float]


class benchmark:
    """Util to benchmark code blocks.
    """

    def __init__(self, msg, fmt='%0.3g'):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        self.start = now()
        return self

    def __exit__(self, *args):
        t = now() - self.start
        log.info(f'{self.msg} in {self.fmt % (t)}s')
        self.time = t


def assign(value: str, subst: dict[str, str] = None, ch: str = None) -> str:
    assert ch is None or is_variable(ch)
    count = 2 + sum(1 for x in (subst or {}).values() if is_variable(x))
    marker = ch or f'A{count}'
    if '::' not in value:
        return marker

    _, kind = value.split('::', maxsplit=1)
    return f'{marker}::{kind}'


def generalize(path: tuple[tuple[str, str, str], ...], subst: dict[str, str]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (k, subst.setdefault(s, assign(s, subst)), subst.setdefault(t, assign(t, subst)))
        for k, s, t in path
    )


def get_confidence(paths: dict[Walk, bool], args: ap.Namespace) -> float:
    total = len(paths)
    confidence = sum(paths.values()) / total
    if args and args.dampen:
        if total not in args.values:
            value = args.dampen * log1p(total)
            args.values[total] = value / (1 + value)
        confidence = args.values[total] * confidence

    return confidence


def get_saturation(current: Report, master: Report) -> float:
    if not master:
        return 0.0

    matches = sum(1 for r in master if r in current)

    return matches / len(master)


def is_cyclic(path: Walk) -> bool:
    assert len(path) >= 2

    origin = path[0][2] if path[0][1] in path[1] else path[0][1]
    destination = path[-1][2] if path[-1][1] in path[-2] else path[-1][1]

    return origin == destination


def is_variable(value: str) -> bool:
    return value[0].isupper() or value[0] == '_'


def float_or_int(astring):
    try:
        result = eval(astring)
        if not isinstance(result, float) and not isinstance(result, int):
            raise ap.ArgumentTypeError(f'float or int expected, but \'{astring}\' found')
    except:
        raise ap.ArgumentTypeError(f'float or int expected, but \'{astring}\' found')

    return result


def md5(filename: str, buf_size: int = 65536) -> str:
    with open(filename, 'rb') as f:
        digest = hl.md5()
        while True:
            data = f.read(buf_size)
            if not data:
                break

            digest.update(data)

        return digest.hexdigest()


def norm(value: str) -> str:
    if '::' not in value:
        return value.lower()

    kind, ident = value.split('::', maxsplit=1)
    return f'{ident.lower()}::{kind}'


class StripFormatter(log.Formatter):
    """ A formatter for logging that strips trailing and ending white spaces from messages wen processing them.
    """

    def format(self, record):
        record.msg = record.msg.strip()
        if record.msg.startswith("- "):
            record.msg = record.msg[2:].strip()

        return super(StripFormatter, self).format(record)


class Graph:

    @staticmethod
    def seed(args: ap.Namespace) -> None:
        log.info(f"\nRandom-state set to {args.rnd_state}")
        rnd.seed(args.rnd_state)
        np.random.seed(args.rnd_state)

    @staticmethod
    def hetionet(args: ap.Namespace) -> None:
        log.info(f"\nDownloading 'Het.io net'...")
        Graph._get_hetionet_edges(args.edges, args.force)
        if args.nodes:
            Graph._get_hetionet_nodes(args.nodes, args.force)

    @staticmethod
    def _get_hetionet_edges(filename: str, force: bool = False) -> None:
        digest = '47f9ee37bccf3cf1dc00e9fd8e4c7056'
        if force or not os.path.exists(filename) or md5(filename) != digest:
            with benchmark('- metadata read'):
                url = 'https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0-metagraph.json'
                with req.get(url) as r:
                    meta = json.loads(r.text)
                convert = {
                    f'{meta['kind_to_abbrev'][s]}{meta['kind_to_abbrev'][r]}{['>', ''][d == 'both']}{meta['kind_to_abbrev'][t]}': r
                    for s, t, r, d in meta['metaedge_tuples']
                }

            with benchmark('- edges downloaded'):
                url = 'https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz'
                with req.get(url) as r:
                    edges = pd.read_csv(io.StringIO(gz.decompress(r.content).decode()), sep='\t').drop_duplicates()
                edges.insert(0, 'source', edges.pop('source').apply(lambda x: norm(x)))
                edges.insert(1, 'target', edges.pop('target').apply(lambda x: norm(x)))
                edges.insert(2, 'kind', edges.pop('metaedge').apply(lambda x: convert[x]))

                edges.to_csv(filename, index=False)

        assert md5(filename) == digest

    @staticmethod
    def _get_hetionet_nodes(filename: str, force: bool = False) -> None:
        digest = '0a73336ad4a3fcfaafddc6c1dd9e0270'
        if force or not os.path.exists(filename) or md5(filename) != digest:
            with benchmark('- nodes downloaded'):
                url = 'https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv'
                with req.get(url) as r:
                    nodes = pd.read_csv(io.StringIO(r.content.decode()), sep='\t').drop_duplicates()
                nodes = nodes.rename(columns={'id': 'name', 'name': 'label'})
                nodes.insert(0, 'name', nodes.pop('name').apply(lambda x: norm(x)))

                nodes.to_csv(filename, index=False)

        assert md5(filename) == digest

    @staticmethod
    def toy() -> Graph:
        edges = pd.DataFrame({
            'source': ['g2', 'g3', 'g4', 'g1', 'g2', 'g3', 'g4', 'g5', 'c1', 'c2', 'c3', 'g2', 'g4', 'g5', 'g1', 'g2',
                       'g3', 'g4', 'g5', 'c1', 'd1', 'c1', 'c3'],
            'target': ['d1', 'd3', 'd2', 'm1', 'm2', 'm1', 'm2', 'm1', 'g1', 'g3', 'g5', 'g1', 'g3', 'g3', 'p1', 'p1',
                       'p1', 'p2', 'p2', 'c2', 'd3', 'd1', 'd3'],
            'kind': ['associates', 'associates', 'associates', 'has_mechanism', 'has_mechanism', 'has_mechanism',
                     'has_mechanism', 'has_mechanism', 'has_target', 'has_target', 'has_target', 'interacts',
                     'interacts', 'interacts', 'participates', 'participates', 'participates', 'participates',
                     'participates', 'resembles', 'resembles', 'treats', 'treats']
        })
        nodes = pd.DataFrame({
            'name': ['c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'g1', 'g2', 'g3', 'g4', 'g5', 'm1', 'm2', 'p1', 'p2'],
            'label': ['compound_1', 'compound_2', 'compound_3', 'disease_1', 'disease_2', 'disease_3', 'gene_1',
                      'gene_2', 'gene_3', 'gene_4', 'gene_5', 'mechanism_1', 'mechanism_2', 'pathway_1', 'pathway_2'],
            'kind': ['Compound', 'Compound', 'Compound', 'Disease', 'Disease', 'Disease', 'Gene', 'Gene', 'Gene',
                     'Gene', 'Gene', 'Mechanism', 'Mechanism', 'Pathway', 'Pathway'],
        })

        return Graph(edges, nodes)

    @staticmethod
    def populate(args: ap.Namespace) -> Graph:
        log.info(f"\nLoading graph...")
        with benchmark('- edges loaded'):
            edges = pd.read_csv(args.edges)
        if args.nodes:
            with benchmark('- nodes loaded'):
                nodes = pd.read_csv(args.nodes)
        else:
            nodes = None

        return Graph(edges, nodes)

    def __init__(self, edges: pd.DataFrame, nodes: pd.DataFrame):
        with benchmark('- graph populated'):
            assert list(edges.columns) == ['source', 'target', 'kind']
            assert list(nodes.columns) == ['name', 'label', 'kind']
            # assert source, target, kind, name do not start with underscore or uppercase
            self.graph = ig.Graph.DataFrame(edges, directed=False, vertices=nodes, use_vids=False)

    def get_examples(self, rel: str) -> pd.DataFrame:
        with benchmark('- examples extracted'):
            pos = pd.DataFrame([{
                'source': self.graph.vs[e.source]['name'],
                'target': self.graph.vs[e.target]['name'],
                'kind': rel
            } for e in self.graph.es.select(kind=rel)])

            srcs = pos[['source']].drop_duplicates()
            tgts = pos[['target']].drop_duplicates()
            result = srcs.merge(tgts, how='cross')
            result = result.loc[result.source != result.target].drop_duplicates()
            result.insert(0, 'kind', rel)
            result = result.merge(pos, how='left', indicator='expected')
            result.insert(0, 'expected', result.pop('expected').apply(lambda x: x == 'both'))

            return result

    def exist(self, triple: Triple) -> bool:
        kind, source, target = triple

        try:
            self.graph.es.find(kind=kind, _source=source, _target=target)
        except ValueError:
            return False
        else:
            return True

    def match(self, triple: Triple) -> Generator[Triple, None, None]:
        kind, source, target = triple

        params = {'kind': kind}
        if not is_variable(source):
            params['_source'] = source
        if not is_variable(target):
            params['_target'] = target

        for edge in self.graph.es.select(**params):
            yield (edge['kind'], self.graph.vs[edge.source]['name'], self.graph.vs[edge.target]['name'])

    def sample(self, examples: pd.DataFrame, length: int) -> tuple[bool, Walk] | None:
        assert list(examples.columns) == ['expected', 'kind', 'source', 'target']

        try:
            expected, kind, source, target = examples.sample().values[0]

            start = rnd.choice([source, target])

            origin = self.graph.vs.find(target if start == source else source)
            previous = [self.graph.vs.find(start)]

            path = [(kind, source, target)]
            for i in range(1, length):
                neighbors = self.graph.neighbors(previous[-1])
                index = rnd.choice(neighbors)
                following = self.graph.vs[index]
                if any(following == p for p in previous) or (i < length - 1 and following == origin):
                    return None

                relation = rnd.choice(self.graph.es.select(_source=previous, _target=following))
                triple = (relation['kind'], previous[-1]['name'], following['name'])
                path.append(triple)
                previous.append(following)
        except ValueError | IndexError:
            return None

        return expected, tuple(path)

    def specify(self, rule: Walk, partial: Walk = None, subst: dict[str, str] = None) -> Generator[Walk, None, None]:
        partial = partial or tuple()
        if len(partial) == len(rule):
            yield partial
        else:
            subst = subst or {}

            kind, source, target = rule[len(partial)]
            atom = kind, subst.get(source, source), subst.get(target, target)
            for triple in self.match(atom):
                if is_variable(atom[1]):
                    subst.setdefault(atom[1], triple[1])
                if is_variable(atom[2]):
                    subst.setdefault(atom[2], triple[2])

                yield from self.specify(rule, (*partial, triple), subst)

    @staticmethod
    def generate(path: Walk) -> Generator[Walk, None, None]:
        origin = path[0][2] if path[0][1] in path[1] else path[0][1]
        pivor = path[0][1] if path[0][1] in path[1] else path[0][2]
        destination = path[-1][2] if path[-1][1] in path[-2] else path[-1][1]

        if origin == destination:  # cyclic
            yield generalize(path, {origin: assign(origin, ch='Y'), pivor: assign(pivor, ch='X')})
            yield generalize(path, {origin: origin, pivor: assign(pivor, ch='X')})
            yield generalize(path, {origin: assign(origin, ch='Y'), pivor: pivor})
        else:
            yield generalize(path, {origin: origin, pivor: assign(pivor, ch='X'), destination: destination})
            yield generalize(path, {origin: origin, pivor: assign(pivor, ch='X')})


class Learn:

    @staticmethod
    def run(graph: Graph, examples: pd.DataFrame, args: ap.Namespace):
        if args.force or not os.path.exists(args.learn):
            rules = Learn.discover(graph, examples, args)

            log.info(f"\nSaving {len(rules)} rule/s to '{args.learn}'...")
            with open(args.learn, "w") as file:
                for rule, confidence in rules.items():
                    print((confidence, rule), file=file)

        else:
            log.info(f"\nLoading rule/s from '{args.learn}'...")
            with open(args.learn, "r") as file:
                rules = {}
                for line in file:
                    if line:
                        confidence, rule = eval(line)
                        rules[rule] = confidence

        return rules

    @staticmethod
    def discover(graph: Graph, examples: pd.DataFrame, args: ap.Namespace) -> Report:
        params = {'min': args.length, 'max closed': args.closed, 'max open': args.open}
        params = ', '.join([f'{k}:{v}' for k, v in params.items() if v])
        log.info(f"\nLearning rules ({params}, and {args.workers} worker/s)...")

        length = args.length
        with mp.Pool(args.workers) as pool:
            log.info(f"\n[{0.0:6.2f}%] {0:,} rule/s found")
            deadline = now() + args.duration
            while True:
                args.saturated = False
                workers = [
                    pool.apply_async(Learn._discover, args=(graph, examples, length, args), callback=Learn._update)
                    for _ in range(args.workers)
                ]
                for w in workers:
                    w.wait()
                if args.saturated:
                    length = length + 1
                    log.info(f"\n- Result set saturated: extending sampling length to '{length}'...")

                time = now()
                elapsed = min(100, 100 - 100 * (deadline - time) / args.duration)
                log.info(f"[{elapsed:6.2f}%] {len(args.rules):,} rule/s found")
                if time >= deadline:
                    break

        rules = {r: get_confidence(p, args) for r, p in args.rules.items()}
        rules = {r: c for r, c in sorted(rules.items(), key=lambda x: (-x[1], str(x[0])))}

        return rules

    @staticmethod
    def _discover(graph: Graph, examples: pd.DataFrame, length: int, args: ap.Namespace) -> Report:
        setup_logging()
        log.debug(f"\nSampling session '{args.number}.{mp.current_process().pid}'...")

        report = {}
        if (not args.open or length < args.open) and (not args.closed or length < args.closed):
            deadline = now() + args.period
            while True:
                found = graph.sample(examples, length)
                if found:
                    expected, path = found
                    for rule in graph.generate(path):
                        limit = args.closed if is_cyclic(rule) else args.open
                        if not limit or len(rule) < limit:
                            report.setdefault(rule, {}).setdefault(path, expected)

                if now() >= deadline:
                    break

        return report

    @staticmethod
    def _update(report: Report) -> None:
        setup_logging()
        log.debug(f"\nCallback session '{args.number}.{mp.current_process().pid}'...")

        args.number += 1
        with open(f"simple_rules_{args.number}.pl", "w") as file:
            for rule, paths in report.items():
                print(rule, file=file)
                for path, correct in paths.items():
                    print("%", (correct, path), file=file)
                print(file=file)

        current = {r: p for r, p in report.items() if get_confidence(p, args) >= args.quality}
        if get_saturation(current, args.rules) >= args.saturation:
            args.saturated = True
        for rule, paths in current.items():
            args.rules.setdefault(rule, {}).update(paths)


class Apply:

    @staticmethod
    def run(graph: Graph, rules: Report, args: ap.Namespace):
        log.info(f"\nApplying rule/s...")
        if args.force or not os.path.exists(args.apply):
            aggregation = Apply.aggregate(graph, rules)
            ranking = Apply.score(aggregation, args)

            log.info(f"\nSaving {len(ranking)} result/s to '{args.apply}'...")
            ranking.to_csv(args.apply, index=False)

        else:
            log.info(f"\nLoading result/s from '{args.apply}'...")
            ranking = pd.read_csv(args.apply)

        return ranking

    @staticmethod
    def aggregate(graph: Graph, rules: Report) -> Aggregation:
        if rules:
            with benchmark("- targets aggregated"):
                data = list(rules.items())
                total = len(data)
                with mp.Pool(args.workers) as pool:
                    log.info(f"\n[{0.0:6.2f}%] {0:,} rule/s processed")
                    while data:
                        workers = [
                            pool.apply_async(Apply._expand, args=(graph, *data.pop(0), args), callback=Apply._update)
                            for _ in range(args.workers) if data
                        ]
                        for w in workers:
                            w.wait()

                        elapsed = 100 * (total - len(data)) / total
                        log.info(f"[{elapsed:6.2f}%] {len(args.rules):,} rule/s processed")

        with benchmark("- scores computed"):
            match args.mode:
                case "maximum":
                    log.info("\nApplying scores using 'maximum' policy...")
                    scores = Counter({t: Apply.maximum(p) for t, p in args.aggregation.items()})
                case "noisy-or":
                    log.info("\nApplying scores using 'noisy-or' policy...")
                    scores = Counter({t: Apply.noisyor(p) for t, p in args.aggregation.items()})
                case _:
                    raise ValueError(f"Unexpected mode: {args.mode}")

        result = pd.DataFrame([
            {"target": t, "score": s, "exist": Graph.exist(t)}
            for t, s in scores.most_common(args.top_k)
        ])

        return result

    @staticmethod
    def _expand(graph: Graph, rule: Walk, confidence: float, args: ap.Namespace) -> tuple[list[Walk], float]:
        return list(graph.specify(rule)), confidence

    @staticmethod
    def _update(stuff: tuple[list[Walk], float]):
        paths, confidence = stuff
        for path in paths:
            args.aggregation.setdefault(path[0], []).append((path, confidence))

    @staticmethod
    def aggregate2(graph: Graph, rules: Report) -> Aggregation:
        result = {}
        if rules:
            with benchmark("- targets aggregated"):
                total = len(rules)
                for i, (rule, confidence) in enumerate(rules.items()):
                    for path in graph.specify(rule):
                        result.setdefault(path[0], {}).setdefault(path, confidence)
                    if i % 100 == 0:
                        log.info(f"  - [{100 * i / total:6.2f}] {i:,} rule/s processed")

        return result

    @staticmethod
    def score(aggregation: Aggregation, args: ap.Namespace) -> pd.DataFrame:
        if aggregation:
            with benchmark("- scores computed"):
                match args.mode:
                    case "maximum":
                        log.info("\nApplying scores using 'maximum' policy...")
                        scores = Counter({t: Apply.maximum(p) for t, p in aggregation.items()})
                    case "noisy-or":
                        log.info("\nApplying scores using 'noisy-or' policy...")
                        scores = Counter({t: Apply.noisyor(p) for t, p in aggregation.items()})
                    case _:
                        raise ValueError(f"Unexpected mode: {args.mode}")

        return result

    @staticmethod
    def maximum(paths: list[tuple[Walk, float]]) -> float:
        return max([e[1] for e in paths], default=0.0)

    @staticmethod
    def noisyor(paths: list[tuple[Walk, float]]) -> float:
        return 1 - prod((1 - e[1]) for e in paths)


def setup_logging() -> None:
    """ Configure `logging` to redirect any log to a file and to the standard output.
    """
    stream_handler = log.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(fmt=log.Formatter(fmt="%(message)s"))
    stream_handler.setLevel(level=log.INFO)

    file_handler = log.FileHandler(
        filename=os.path.join(os.getcwd(), "pyLomma.log"),
        encoding="UTF-8",
        mode="w",
    )
    file_handler.setFormatter(fmt=StripFormatter(
        fmt="%(asctime)s  %(levelname)-8s  p%(process)s@%(filename)s:%(lineno)d - %(message)s"))
    file_handler.setLevel(level=log.DEBUG)

    logger = log.getLogger()
    logger.addHandler(hdlr=stream_handler)
    logger.addHandler(hdlr=file_handler)
    logger.setLevel(level=log.DEBUG)


def header() -> None:
    log.info(f"pyLomma v0.1.0")


def parse_args(default: Sequence[str] = None) -> ap.Namespace:
    """Parse arguments from the command line and return them.

    If a `default` sequence of strings is given, the arguments are parsed from there.

    :param default: the optional sequence of strings from where to parse the arguments
    :return: the parsed arguments
    """
    parser = ap.ArgumentParser(description=f"Python implementation of AnyBURL "
                                           f"(https://web.informatik.uni-mannheim.de/AnyBURL/).")

    parser.add_argument("--apply", "-a", required=True, type=str,
                        help="use this file for ranking")
    parser.add_argument("--closed", "-c", required=False, default=None, type=int,
                        help="limit the length of closed (cyclic) rules to this value")
    parser.add_argument("--duration", "-d", required=True, default=20, type=float,
                        help="the desired duration in seconds of learning")
    parser.add_argument("--edges", "-e", required=True, type=str,
                        help="use this file for graph edges", metavar="EDGES.CSV")
    parser.add_argument("--force", "-f", required=False, default=False, action='store_true',
                        help="force recomputing results")
    parser.add_argument("--hetionet", "-i", required=False, default=False, action='store_true',
                        help="download Het.io net to given EDGES.CSV (and NODES.CSV, if given)")
    parser.add_argument("--length", "--g", required=False, default=2, type=int,
                        help="use this value to set the minimum length of paths to sample")
    parser.add_argument("--learn", "-l", required=True, type=str,
                        help="use this file for rules")
    parser.add_argument("--mode", "-m", required=False, type=str, choices=["maximum", "noisy-or"], default='maximum',
                        help="use this mode to score the aggregated target confidence")
    parser.add_argument("--nodes", "-n", required=False, default=None, type=str,
                        help="use this file for graph nodes", metavar="NODES.CSV")
    parser.add_argument("--open", "-o", required=False, default=None, type=int,
                        help="limit the length of open (acyclic) rules to this value")
    parser.add_argument("--period", "-p", required=True, type=float,
                        help="the desired duration in seconds of a period of learning")
    parser.add_argument("--quality", "-q", required=True, type=float,
                        help="the desired value for rules quality")
    parser.add_argument("--rnd-state", "-r", required=False, type=int, default=None,
                        help="use this seed to set the random state", metavar="SEED")
    parser.add_argument("--saturation", "-s", required=True, type=float,
                        help="the desired value for rules saturation")
    parser.add_argument("--target", "-t", required=False, default=None, type=str,
                        help="learn this kind of relation")
    parser.add_argument("--evaluate", "-v", required=False, default=None, type=str,
                        help="use this file for ROC curve results", metavar="NODES.CSV")
    parser.add_argument("--workers", "-w", required=False, default=floor((3 * mp.cpu_count() / 4)), type=int,
                        help="use this number of workers for sampling the graph in parallel")
    parser.add_argument("--split", "-y", required=False, default=None, type=float_or_int,
                        help="split the examples in train and test by this factor")
    parser.add_argument("--dampen", "-z", required=False, default=None, type=float,
                        help="dampen the confidence of the rules by applying this factor")

    args = parser.parse_args(args=default)
    log.debug(f"Parsed args: {args}")
    setattr(args, 'aggregation', {})
    setattr(args, 'number', 0)
    setattr(args, 'rules', {})
    setattr(args, 'saturated', False)
    setattr(args, 'values', {})

    return args


if __name__ == '__main__':
    setup_logging()
    header()
    args = parse_args([
        "-a", "ranking.csv",
        "-d", "30",  # "2400",
        # "-f",
        "-e", "hetionet-v1.0-edges.csv",
        # "-i",
        "-l", "rules.pl",
        "-n", "hetionet-v1.0-nodes.csv",
        "-p", "10",  # "30",
        "-q", "0.25",
        "-r", "42",
        "-s", "0.75",
        "-t", "treats",
        "-w", "12",
        "-y", "1/3",
        "-z", "1.7",
    ])

    mp.freeze_support()
    if args.rnd_state:
        Graph.seed(args)

    if args.hetionet:
        Graph.hetionet(args)

    g = Graph.populate(args)
    ex = g.get_examples(args.target)

    if args.split or args.evaluate:
        with benchmark('- examples split'):
            train, test = train_test_split(ex, test_size=args.split, shuffle=True, stratify=ex.expected)
    else:
        train, test = ex, None

    rules = Learn.run(g, train, args)

    ranking = Apply.run(g, rules, args)

    print(ranking)

    log.info('\nDone.')
