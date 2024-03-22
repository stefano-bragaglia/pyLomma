import argparse as ap
import logging as log
import multiprocessing as mp
import os
from timeit import default_timer as now

import pandas as pd
from sklearn.model_selection import train_test_split

from fast import get_confidence
from fast import get_saturation
from fast import Graph
from fast import is_cyclic
from fast import setup_logging

Report = dict


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


if __name__ == '__main__':
    args = ap.Namespace(
        closed=None,
        dampen=None,
        duration=30,
        force=True,
        learn='temp_rules.pl',
        length=2,
        number=0,
        open=None,
        period=10,
        quality=0.75,
        rnd_state=42,
        rules={},
        saturated=False,
        saturation=0.5,
        split=1/3,
        workers=2,
    )
    setup_logging()
    Graph.seed(args)

    graph = Graph.toy()
    examples = graph.get_examples('treats')
    print(examples)
    # train, test = train_test_split(examples, test_size=args.split, shuffle=True, stratify=examples.expected)
    # print(train)
    # print(test)
    result = Learn.run(graph, examples, args)

    print(result)

    print('Done.')
