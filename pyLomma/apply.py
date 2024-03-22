import argparse as ap
import logging as log
import multiprocessing as mp
import os
from collections import Counter
from math import prod

import pandas as pd

from fast import benchmark
from fast import Graph

Triple = tuple[str, str, str]
Walk = tuple[Triple, ...]

Report = dict
Aggregation = dict


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


if __name__ == '__main__':
    args = ap.Namespace(
        workers=2,
    )

    rules = {
        (('treats', 'X', 'Y'), ('has_target', 'X', 'A4'), ('interacts', 'A4', 'A5'), ('associates', 'A5', 'Y')): 1.0,
        (('treats', 'X', 'Y'), ('has_target', 'X', 'A4'), ('interacts', 'A4', 'A5'), ('treats', 'A5', 'Y')): 1.0,
        (('treats', 'X', 'Y'), ('treats', 'X', 'Y')): 1.0,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'), ('has_mechanism', 'A4', 'g5')): 1.0,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('associates', 'A4', 'd1')): 1.0,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('has_mechanism', 'A4', 'm2')): 1.0,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('treats', 'A4', 'd1')): 1.0,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'g2')): 1.0, (
        ('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'A4'), ('interacts', 'A4', 'A5')): 1.0,
        (
        ('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'A4'), ('interacts', 'A4', 'g2')): 1.0,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'A4'),
         ('participates', 'A4', 'g2')): 1.0, (
        ('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'A4'),
        ('participates', 'A4', 'g3')): 1.0,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'p1')): 1.0,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'g1')): 1.0, (('treats', 'X', 'd1'), ('resembles', 'X', 'A3')): 1.0,
        (('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'A4')): 1.0,
        (('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'A4'), ('associates', 'A4', 'A5')): 1.0,
        (('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'A4'), ('associates', 'A4', 'd3')): 1.0,
        (('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'A4'),
         ('has_mechanism', 'A4', 'A5')): 1.0, (
        ('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'm1')): 1.0,
        (('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'A4'), ('interacts', 'A4', 'A5')): 1.0,
        (('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'A4'), ('interacts', 'A4', 'g4')): 1.0,
        (('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'A4'), ('interacts', 'A4', 'g5')): 1.0, (
        ('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'A4'), ('participates', 'A4', 'A5')): 1.0,
        (
        ('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'A4'), ('participates', 'A4', 'p1')): 1.0,
        (('treats', 'X', 'd1'), ('resembles', 'X', 'A3'), ('has_target', 'A3', 'g3')): 1.0,
        (('treats', 'X', 'd1'), ('resembles', 'X', 'c2')): 1.0, (('treats', 'X', 'd1'), ('treats', 'X', 'd1')): 1.0,
        (('treats', 'X', 'd1'), ('treats', 'd1', 'X')): 1.0, (
        ('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'g1')): 1.0, (
        ('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('interacts', 'A4', 'A5')): 1.0, (
        ('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('interacts', 'A4', 'g3')): 1.0,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('associates', 'A4', 'd3')): 1.0,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'),
         ('has_mechanism', 'A4', 'm1')): 1.0,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('has_target', 'A4', 'A5')): 1.0,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('has_target', 'A4', 'c2')): 1.0,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('interacts', 'A4', 'A5')): 1.0,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('interacts', 'A4', 'g4')): 1.0,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('treats', 'A4', 'd3')): 1.0,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'g3')): 1.0, (
        ('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'A4'),
        ('participates', 'A4', 'g4')): 1.0,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'p2')): 1.0,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'g5')): 1.0, (('treats', 'X', 'd3'), ('treats', 'X', 'd3')): 1.0,
        (('treats', 'X', 'd3'), ('treats', 'd3', 'X')): 1.0,
        (('treats', 'Y', 'X'), ('associates', 'X', 'A4'), ('interacts', 'A4', 'A5'), ('has_target', 'A5', 'Y')): 1.0,
        (('treats', 'Y', 'X'), ('associates', 'X', 'A4'), ('interacts', 'A4', 'A5'), ('treats', 'A5', 'Y')): 1.0,
        (('treats', 'Y', 'd1'), ('associates', 'd1', 'A3'), ('interacts', 'A3', 'A4'), ('has_target', 'A4', 'Y')): 1.0,
        (('treats', 'Y', 'd1'), ('associates', 'd1', 'A3'), ('interacts', 'A3', 'A4'), ('treats', 'A4', 'Y')): 1.0,
        (('treats', 'Y', 'd3'), ('associates', 'd3', 'A3'), ('interacts', 'A3', 'A4'), ('has_target', 'A4', 'Y')): 1.0,
        (('treats', 'Y', 'd3'), ('associates', 'd3', 'A3'), ('interacts', 'A3', 'A4'), ('treats', 'A4', 'Y')): 1.0, (
        ('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'g4')): 1.0,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'm2')): 1.0,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('has_target', 'A4', 'c1')): 1.0,
        (
        ('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('participates', 'A4', 'p1')): 1.0,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('treats', 'A4', 'c1')): 1.0,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'g1')): 1.0, (
        ('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'A4'), ('interacts', 'A4', 'A5')): 1.0,
        (
        ('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'A4'), ('interacts', 'A4', 'g1')): 1.0,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'A4'),
         ('participates', 'A4', 'g3')): 1.0, (('treats', 'c1', 'X'), ('associates', 'X', 'g2')): 1.0, (
        ('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'm1')): 1.0,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'), ('has_target', 'A4', 'A5')): 1.0,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'), ('has_target', 'A4', 'c2')): 1.0,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'), ('interacts', 'A4', 'g4')): 1.0,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'), ('interacts', 'A4', 'g5')): 1.0,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'g3')): 1.0,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('treats', 'A3', 'A4')): 1.0,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('treats', 'A3', 'A4'), ('has_target', 'A4', 'A5')): 1.0,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('treats', 'A3', 'A4'), ('has_target', 'A4', 'g5')): 1.0,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('treats', 'A3', 'c3')): 1.0,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'd3')): 1.0,
        (('treats', 'c1', 'Y'), ('has_target', 'c1', 'A3'), ('interacts', 'A3', 'A4'), ('associates', 'A4', 'Y')): 1.0,
        (('treats', 'c1', 'Y'), ('has_target', 'c1', 'A3'), ('interacts', 'A3', 'A4'), ('treats', 'A4', 'Y')): 1.0,
        (('treats', 'c1', 'Y'), ('treats', 'c1', 'Y')): 1.0, (('treats', 'c1', 'd1'), ('treats', 'd1', 'c1')): 1.0, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'g1')): 1.0, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'g5')): 1.0, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('interacts', 'A4', 'A5')): 1.0, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('interacts', 'A4', 'g5')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'm1')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_target', 'A3', 'A4')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_target', 'A3', 'A4'), ('resembles', 'A4', 'A5')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_target', 'A3', 'A4'), ('resembles', 'A4', 'c1')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_target', 'A3', 'c2')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('associates', 'A4', 'A5')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('associates', 'A4', 'd2')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'),
         ('has_mechanism', 'A4', 'm2')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('has_target', 'A4', 'c3')): 1.0,
        (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('participates', 'A4', 'p2')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('treats', 'A4', 'c3')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'g4')): 1.0,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'g5')): 1.0, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'A4'),
        ('participates', 'A4', 'g2')): 1.0, (('treats', 'c3', 'X'), ('associates', 'X', 'g3')): 1.0, (
        ('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'm2')): 1.0,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'), ('interacts', 'A4', 'g1')): 1.0,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'g2')): 1.0,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('treats', 'A3', 'A4')): 1.0,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('treats', 'A3', 'A4'), ('has_target', 'A4', 'A5')): 1.0,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('treats', 'A3', 'A4'), ('has_target', 'A4', 'g1')): 1.0,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('treats', 'A3', 'A4'), ('resembles', 'A4', 'A5')): 1.0,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('treats', 'A3', 'A4'), ('resembles', 'A4', 'c2')): 1.0,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('treats', 'A3', 'c1')): 1.0,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'd1')): 1.0,
        (('treats', 'c3', 'Y'), ('has_target', 'c3', 'A3'), ('interacts', 'A3', 'A4'), ('associates', 'A4', 'Y')): 1.0,
        (('treats', 'c3', 'Y'), ('has_target', 'c3', 'A3'), ('interacts', 'A3', 'A4'), ('treats', 'A4', 'Y')): 1.0,
        (('treats', 'c3', 'Y'), ('treats', 'c3', 'Y')): 1.0, (('treats', 'c3', 'd3'), ('treats', 'd3', 'c3')): 1.0, (
        ('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'A4'),
        ('participates', 'A4', 'A5')): 0.6666666666666666, (
        ('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'),
        ('interacts', 'A4', 'A5')): 0.6666666666666666, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'A5')): 0.6666666666666666,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4')): 0.6666666666666666, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'A5')): 0.6666666666666666, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'),
        ('participates', 'A4', 'A5')): 0.6666666666666666, (('treats', 'X', 'd1'), ('has_target', 'X', 'A3')): 0.5,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'A4')): 0.5, (
        ('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'A5')): 0.5, (
        ('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'g3')): 0.5,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'm1')): 0.5,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4')): 0.5, (
        ('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'A5')): 0.5, (
        ('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('participates', 'A4', 'A5')): 0.5,
        (
        ('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('participates', 'A4', 'p1')): 0.5,
        (('treats', 'X', 'd1'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'A4')): 0.5,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3')): 0.5,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'A4')): 0.5, (
        ('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'A5')): 0.5, (
        ('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'g3')): 0.5,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('has_mechanism', 'A3', 'm1')): 0.5,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4')): 0.5, (
        ('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'A5')): 0.5, (
        ('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('participates', 'A4', 'A5')): 0.5,
        (
        ('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('participates', 'A4', 'p1')): 0.5,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'A4')): 0.5,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3')): 0.5,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'A4')): 0.5, (
        ('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'm1')): 0.5,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'A4')): 0.5, (
        ('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'A4'),
        ('participates', 'A4', 'A5')): 0.5, (
        ('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'A4'),
        ('participates', 'A4', 'g1')): 0.5,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'p1')): 0.5,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3')): 0.5,
        (('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4')): 0.5, (
        ('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'A5')): 0.5, (
        ('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'), ('participates', 'A4', 'A5')): 0.5,
        (
        ('treats', 'c1', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'), ('participates', 'A4', 'p1')): 0.5,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3')): 0.5,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'A4')): 0.5, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'm1')): 0.5,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'A4')): 0.5, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'A4'),
        ('participates', 'A4', 'A5')): 0.5, (
        ('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'A4'),
        ('participates', 'A4', 'g1')): 0.5,
        (('treats', 'c3', 'X'), ('associates', 'X', 'A3'), ('participates', 'A3', 'p1')): 0.5,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3')): 0.5,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4')): 0.5, (
        ('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'A5')): 0.5, (
        ('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'), ('participates', 'A4', 'A5')): 0.5,
        (
        ('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'), ('participates', 'A4', 'p1')): 0.5,
        (('treats', 'X', 'd3'), ('has_target', 'X', 'A3'), ('participates', 'A3', 'A4'),
         ('participates', 'A4', 'A5')): 0.3333333333333333, (
        ('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('has_mechanism', 'A3', 'A4'),
        ('has_mechanism', 'A4', 'A5')): 0.3333333333333333,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4')): 0.3333333333333333,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('has_mechanism', 'A4', 'A5')): 0.3333333333333333,
        (('treats', 'c1', 'X'), ('associates', 'X', 'A3'), ('interacts', 'A3', 'A4'), ('participates', 'A4', 'A5')): 0.3333333333333333,
        (('treats', 'c3', 'X'), ('resembles', 'X', 'A3'), ('associates', 'A3', 'A4'), ('interacts', 'A4', 'A5')): 0.3333333333333333}

    print('Done.')
