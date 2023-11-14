import os
import torch
import numpy as np
from abc import ABC
from stl import mesh
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class GraphImporter(InMemoryDataset, ABC):
    def __init__(self, raw_data_root, root, transform=None):
        self.data_list = []
        self.raw_data_root = raw_data_root
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def processed_file_names(self):
        return 'data.pt'

    @staticmethod
    def cad_graph_conversion(cad_directory, file_labels):
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
        x = torch.tensor(x / np.array([10, 10, 10])).float()
        edge_index = torch.tensor(edge_index)
        label_array = np.zeros(24)
        for label in file_labels:
            label_array[label] = 1

        graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=torch.tensor(label_array))

        return graph

    def process(self):
        for root, dirs, files in os.walk(self.raw_data_root):
            for file in files:
                file_labels = []

                if file.lower().endswith('.csv'):
                    with open(f'{root}/{file}', 'r', encoding='utf-8') as f:

                        for line in f.readlines():
                            file_labels.append(int(line.split(",")[-1]))

                        file_name = str(file).replace('.csv', '.stl')
                        self.data_list.append(self.cad_graph_conversion(root + '/' + file_name, file_labels))
                        print(file_name)
        torch.save(self.collate(self.data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
