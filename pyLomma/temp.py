import csv

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import patches as ptc
from main import PathParser
from main import Triple

# TODO: convert to iGraph, if possible!
def plot(graph: nx.Graph, title: str, filename: str = None) -> None:
    plt.figure()
    plt.title(title)

    pos = nx.kamada_kawai_layout(graph)

    max_edge_usage = max((graph.get_edge_data(*e).get('usages', 1) for e in graph.edges), default=1)
    max_node_usage = max((graph.nodes[n].get('usages', 1) for n in graph.nodes), default=1)

    edge_width = [1.0 + 2.0 / max_edge_usage * graph.get_edge_data(*e)['usages'] for e in graph.edges]
    nx.draw_networkx_edges(graph, pos, width=edge_width, alpha=0.3)

    edge_labels = {}
    for s, t, r in graph.edges:
        edge_labels.setdefault((s, t), set()).add(r)
    edge_labels = {c: "\n".join(sorted(r)) for c, r in edge_labels.items()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)

    types = ['Compounds', 'Diseases', 'Genes', 'Mechanisms of Action', 'Pathways']
    node_types = sorted({n.split("__", maxsplit=1)[0] for n in graph.nodes})
    colors = ["#50514F", "#F25F5C", "#FFE066", "#247BA0", "#2C6E49"]

    node_size = [int(300 + 500 / max_node_usage * graph.nodes[n]['usages']) for n in graph.nodes]
    node_color = [colors[node_types.index(n[0]) % len(colors)] for n in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color)

    labels = {n: n if node_types.index(n[0]) % len(colors) == 0 else "" for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_color="w")
    labels = {n: n if node_types.index(n[0]) % len(colors) != 0 else "" for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_color="k")

    handles = [ptc.Patch(color=colors[i], label=types[i]) for i, _ in enumerate(node_types)]
    plt.legend(handles=handles)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    with open("../data/graph.csv", "r") as file:
        triples = []
        G = nx.MultiGraph()
        for atom in csv.DictReader(file):
            triple = Triple(**atom)
            if triple not in triples:
                triples.append(triple)
            for name in (atom['source'], atom['target']):
                if G.has_node(name):
                    usages = G.nodes[name]['usages'] + 1
                else:
                    usages = 1
                G.add_node(name, usages=usages)
            if G.has_edge(atom['source'], atom['target'], key=atom['relation']):
                usages = G.get_edge_data(atom['source'], atom['target'], key=atom['relation'])['usages'] + 1
            else:
                usages = 1
            G.add_edge(atom['source'], atom['target'], key=atom['relation'], usages=usages)

        plot(G, "Toy Graph", f"../data/toy_graph.png")

    aggregation = {}
    parser = PathParser()
    with open('paths.pl', 'r') as file:
        for line in file:
            path = parser.parse(line)
            aggregation.setdefault(path.head, []).append(path)

    for target, paths in aggregation.items():
        G = nx.MultiGraph()
        for path in paths:
            for atom in path.body:
                for name in (atom.source, atom.target):
                    if G.has_node(name):
                        usages = G.nodes[name]['usages'] + 1
                    else:
                        usages = 1
                    G.add_node(name, usages=usages)
                if G.has_edge(atom.source, atom.target, key=atom.relation):
                    usages = G.get_edge_data(atom.source, atom.target, key=atom.relation)['usages'] + 1
                else:
                    usages = 1
                G.add_edge(atom.source, atom.target, key=atom.relation, usages=usages)

        plot(G, f"{target} {"(novel)" if target not in triples else ""}", f"../data/{target.source}_{target.relation}_{target.target}.png")

    print('Done.')
