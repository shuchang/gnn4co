import numpy as np
import networkx as nx
import torch_geometric
from infrastructure import pytorch_utils as ptu


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

    def reset(self):
        self.g = nx.barabasi_albert_graph(self.n_nodes, self.m_edges)

    def get_laplacian(self):
        # adj for training, laplacian for max cut
        laplacian_matrix = nx.laplacian_matrix(self.g).toarray()
        return laplacian_matrix

    def get(self, state):
        data = torch_geometric.utils.from_networkx(self.g).to(ptu.device)
        data.x = ptu.from_numpy(state)
        return data


############################################
############################################

def normalize(data, mean, std, eps=1e-8):
    return (data - mean)/(std + eps)

def unnormalize(data, mean, std):
    return data*std + mean