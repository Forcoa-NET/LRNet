import networkx as nx
import math
from sklearn.metrics.pairwise import rbf_kernel

tolerance = 1e-6

def get_neighbors(S, o):
    for index, similarity in enumerate(S[o]):
        if index == o or similarity < tolerance:
            continue
        yield similarity


def get_neighbors_i(S, o):
    for index, similarity in enumerate(S[o]):
        if index == o or similarity < tolerance:
            continue
        yield index, similarity


def calculate_representativeness(S, min_neighbors, reduction):
    objects = list(range(len(S)))
    # Local degree
    d = [sum(1 for _ in get_neighbors(S, o)) for o in objects]
    # Local significance
    g = [0] * len(objects)
    max_similarity = [max(get_neighbors(S, o)) for o in objects]
    for o in objects:
        for i, sim in get_neighbors_i(S, o):
            if math.isclose(sim, max_similarity[o], rel_tol=tolerance):
                g[i] += 1
    # x-representativeness base
    b = [((1 + d[o]) ** (1 / g[o]) if g[o] > 0 else 0) for o in objects]
    # local representativeness
    lr = [(1 / b[o] if g[o] > 0 else 0) for o in objects]

    for o in objects:
        k = lr[o] * d[o]
        k *= reduction
        k = max(min_neighbors, int(round(k)))

        neighbors = sorted(get_neighbors_i(S, o), key=lambda x: x[1], reverse=True)
        last = neighbors[k-1][1]
        # include neighbors with same similarity for deterministic behaviour
        while math.isclose(last, neighbors[k][1], rel_tol=tolerance):
            k += 1
        yield o, [x[0] for x in neighbors[:k]]


def LRNet(D, min_neighbors=1, reduction=1.0):
    S = rbf_kernel(D.drop('variety', axis=1), gamma=1)
    R = calculate_representativeness(S, min_neighbors, reduction)
    G = nx.Graph()

    for i, o in D.iterrows():
        # o=o) # TODO: mapping of object -> node that is passed to the add_node
        G.add_node(i, label=o.variety)

    for o_i, neighbors in R:
        for o_j in neighbors:
            G.add_edge(o_i, o_j)

    return G


if __name__ == "__main__":
    import pandas as pd
    D = pd.read_csv('iris.csv', ';', nrows=150)
    label = 'variety'
    D[label] = D[label].astype('category')

    G = LRNet(D)
