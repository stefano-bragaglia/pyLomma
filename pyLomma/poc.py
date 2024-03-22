import argparse as ap

Triple = tuple[str, str, str]
Edge = tuple[str, str, str]
Node = tuple[str, str, str]
Path = tuple[bool, tuple[tuple[str, str, str], ...]]
Rule = tuple[bool, tuple[tuple[str, str, str], ...]]
Graph = tuple[list[Edge], list[Node]]
Example = tuple[bool, str, str, str]
Ranking = dict[str, float]


def load(file: str) -> list[tuple[str, str, str]]:
    pass


def save(data: any, file: str) -> None:
    pass


def get_graph(edges: list[Edge], nodes: list[Node] = None) -> Graph:
    # nodes are optional
    pass


def get_examples(graph: any, relation: str = None) -> list[Example]:
    if relation is None:  # all graph
        pass
    else:  # specific relation
        pass


def split(examples: list[Example], perc: float) -> tuple[list[Example], list[Example]]:
    pass


def learn(graph: Graph, examples: list[Example]) -> tuple[list[Path], list[Rule]]:
    pass


def predict(graph: Graph, rules: list[Rule]) -> Ranking:
    pass


def evaluate(ranking: Ranking, examples: list[Example], file: str) -> None:
    pass


def explain(graph: Graph, rules: list[Rule], target: Triple, file: str) -> None:
    pass


def as_file(triple: Triple) -> str:
    pass


def random_state(seed: float) -> None:
    pass


def setup() -> None:
    pass


def parse() -> ap.Namespace:
    return ap.Namespace(edges='edges.csv', nodes=None, relation=None)


if __name__ == '__main__':
    args = parse()

    setup()
    if args.seed:

        random_state(args.seed)

    edges = load(args.edges)
    nodes = load(args.nodes) if args.nodes else None
    graph = get_graph(edges, nodes)
    if args.relation:
        examples = get_examples(graph, args.relation)
    else:
        examples = get_examples(graph)
    if args.perc:
        train, test = split(examples, args.perc)
    else:
        train, test = examples, None

    if args.image and test is None:
        raise ValueError()

    if args.rules:
        rules = load(args.rules)
    else:
        paths, rules = learn(graph, train)
        if args.paths:
            save(paths, args.paths)

    ranking = predict(graph, rules)
    if args.ranking:  # required? what if I only want to explain? then not, but if I want to evaluate, then yes.
        save(ranking, args.ranking)

    if args.image and test:
        evaluate(ranking, test, args.image)

    if args.targets:
        for target in args.target:
            explain(graph, rules, target, as_file(target))

    print('Done.')
