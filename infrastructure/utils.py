import numpy as np
import networkx as nx


class GraphGenerator(object):

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes

    def get(self):
        """Get the adjacent matrix of the graph"""
        raise NotImplementedError


class RandomGraphGenerator(GraphGenerator):

    def __init__(self, n_nodes):
        super().__init__(n_nodes)

    def get(self):
        density = np.random.uniform()
        adj_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if np.random.uniform() < density:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        return adj_matrix


class BarabasiAlbertGraphGenerator(GraphGenerator):

    def __init__(self, n_nodes, m_edges):
        super().__init__(n_nodes)
        self.m_edges = m_edges

    def get_laplacian(self):
        # adj for training, laplacian for max cut
        g = nx.barabasi_albert_graph(self.n_nodes, self.m_edges)
        laplacian_matrix = nx.laplacian_matrix(g).toarray()
        return laplacian_matrix