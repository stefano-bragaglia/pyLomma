import networkx as nx
from matplotlib import pyplot as plt

from main import PathParser

if __name__ == '__main__':
    aggregation = {}
    parser = PathParser()
    with open('paths.pl', 'r') as file:
        for line in file:
            path = parser.parse(line)
            aggregation.setdefault(path.head, []).append(path)

    # print(aggregation)

    for target, paths in aggregation.items():
        G = nx.MultiGraph()
        for path in paths:
            for atom in path.body:
                if G.has_edge(atom.source, atom.target, key=atom.relation):
                    usages = G.get_edge_data(atom.source, atom.target, key=atom.relation)['usages']
                else:
                    usages = 1
                G.add_edge(atom.source, atom.target, key=atom.relation, usages=usages)

        fig, ax = plt.subplots(figsize=(12, 12))

        pos = nx.kamada_kawai_layout(G)

        edgewidth = [len(G.get_edge_data(u, v)) for u, v in G.edges()]

        nx.draw_networkx_edges(G, pos, alpha=0.3, width=edgewidth, edge_color="m")
        nx.draw_networkx_nodes(G, pos,  node_color="#210070", alpha=0.9)  # node_size=nodesize,
        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_labels(G, pos, font_size=14, bbox=label_options)

        ax.text(
            0.80,
            0.10,
            repr(target),
            # horizontalalignment="center",
            # transform=ax.transAxes,
            # fontdict=font,
        )

        # nx.draw(G, pos=pos)
        plt.show()
        # print(G)
        # break


    print('Done.')
