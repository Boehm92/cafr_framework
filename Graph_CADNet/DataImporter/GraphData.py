import torch
import numpy as np
from stl import mesh
from torch_geometric.data import Data


def create(cad_directory, graph_label):
    # numpy-stl method to import an .STL file as mesh object
    m = mesh.Mesh.from_file(cad_directory)

    # Extract all the unique vectors from m.vectors
    x = np.array(np.unique(m.vectors.reshape([int(m.vectors.size / 3), 3]), axis=0))

    # create an edge list from mesh object
    edge_index = []
    for facet in m.vectors:
        index_list = []
        for vector in facet:
            for count, features in enumerate(x):
                if np.array(vector == features).all():
                    index_list.append(count)
        edge_index.append([index_list[0], index_list[1]])
        edge_index.append([index_list[1], index_list[0]])
        edge_index.append([index_list[1], index_list[2]])
        edge_index.append([index_list[2], index_list[1]])
        edge_index.append([index_list[2], index_list[0]])
        edge_index.append([index_list[0], index_list[2]])

    # create graph objects with the x and edge_index list
    x = torch.tensor(x)
    edge_index = torch.tensor(edge_index)
    y = torch.tensor(graph_label)

    graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)

    return graph
