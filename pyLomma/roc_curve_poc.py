from argparse import Namespace
from random import choice
from random import randint
from random import random
from random import seed

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


class MockClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y=None):
        self.X = X
        self.y = y

    def predict(self, X, y=None):
        return X.score


def generate(args: Namespace) -> list[dict[str, any]]:
    if args.random_state:
        seed(args.random_state)

    rows = {}
    while len(rows) < 2 * args.positives:
        source = randint(1, args.sources)
        target = randint(1, args.targets)
        atom = f"{args.relation}(c{source},d{target})"

        score = random()
        rows.setdefault(atom, {
            "atom": atom,
            "score": score,
            "target": max(0.0, min(1.0, score + args.factor * (2 * random() - 1) >= args.rate))
            if args.rate else choice([False, True])
        })

    return list(rows.values())


if __name__ == '__main__':
    experiments = {
        "random": Namespace(
            random_state=42,
            positives=755,
            sources=1552,
            targets=137,
            relation="treats",
            factor=0.1,
            rate=None,
        ),
        "0.35 +/- 0.1": Namespace(
            random_state=42,
            positives=755,
            sources=1552,
            targets=137,
            relation="treats",
            factor=0.1,
            rate=0.35,
        ),
        "0.5 +/- 0.75": Namespace(
            random_state=42,
            positives=755,
            sources=1552,
            targets=137,
            relation="treats",
            factor=0.75,
            rate=0.5,
        ),
    }

    fpr, tpr, roc_auc = {}, {}, {}
    for name, args in experiments.items():
        X = pd.DataFrame(generate(args))
        y = X.pop("target")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 / 3, random_state=42, shuffle=True, stratify=y)

        cls = MockClassifier()
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)

        fpr[name], tpr[name], _thresholds = roc_curve(y_test, y_pred)
        roc_auc[name] = auc(fpr[name], tpr[name])

    plt.figure()
    lw = 2
    for name in roc_auc:
        f, t, r = fpr[name], tpr[name], roc_auc[name]
        plt.plot(f, t, lw=lw, label=f"ROC curve '{name}' (area = {r:0.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.00])
    plt.ylim([0.00, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

    # TODO: with AnyBURL, instead of letting the algorithm to generate random head (even if by giving relation),
    # TODO: produce your own list (both positive and negative examples -- it's a list of samples) and use that
    # TODO: to initiate random walks.

    # TODO: how long it takes to read csv files in triples?

    print('Done.')
