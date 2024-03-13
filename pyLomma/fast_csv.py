import csv
from timeit import default_timer as timer
from typing import NamedTuple

import pandas as pd

Triple = NamedTuple("Triple", source=str, relation=str, target=str)


class benchmark:
    """Util to benchmark code blocks.
    """

    def __init__(self, msg, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start
        print(("%s: " + self.fmt + " seconds") % (self.msg, t))
        self.time = t


if __name__ == '__main__':
    with benchmark("  .CSV as text file"):
        with open("../data/hetionet.csv", "r") as file:
            triples = [Triple(*line.split(",")) for line in file][1:]
            print(f"{len(triples):,} triple/s read")

    with benchmark("  .CSV as DictReader file"):
        with open("../data/hetionet.csv", "r") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == ["source", "relation", "target"]
            triples = [Triple(*r) for r in reader]
            print(f"{len(triples):,} triple/s read")

    with benchmark("  .CSV as pandas file"):
        triples = [Triple(*v) for v in pd.read_csv("../data/hetionet.csv").values]
        print(f"{len(triples):,} triple/s read")

    with benchmark("  .CSV as pure pandas file"):
        df = pd.read_csv("../data/hetionet.csv")
        print(f"{len(df):,} triple/s read")

    print("Pure pandas is **by far** the fastest option.")
    print("Crucial with larger knowledge graphs.")

    print('Done.')
