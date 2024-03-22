import igraph as ig
import pandas as pd

from fast import Graph

if __name__ == '__main__':
    g = Graph.toy()
    print(g.graph)
    
    graph = g.graph

    # is_var = lambda x: x[0].isupper()
    # for k, s, t in [
    #     ('treats', 'XX', 'YY'),
    #     ('treats', 'XX', 'd3'),
    #     ('treats', 'c2', 'YY'),
    #     ('treats', 'c1', 'd1'),
    # ]:
    #     print('>', k, s, t)
    #     params = {'kind': k}
    #     if not is_var(s):
    #         params['_source'] = s
    #     if not is_var(t):
    #         params['_target'] = t
    #
    #     for e in graph.es.select(**params):
    #         print(e['kind'], graph.vs[e.source]['name'], graph.vs[e.target]['name'])
    #     print()

    for k, s, t in [
        ('treats', 'c1', 'd1'),
        ('treats', 'c2', 'd2'),
    ]:
        print('>', k, s, t)
        try:
            # result = graph.es.find(kind=k, _source=s, _target=t)
            result = graph.es.select(kind=k, _source=s, _target=t)
        except ValueError:
            print('not found')
        else:
            print('found')
            for edge in result:
                print(' -', edge['kind'], graph.vs[edge.source]['name'], graph.vs[edge.target]['name'])
        print()

    print('Done.')
