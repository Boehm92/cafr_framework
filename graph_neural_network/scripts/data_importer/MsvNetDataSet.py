import os
import torch
from torch_geometric.data import InMemoryDataset
from graph_neural_network.scripts.data_importer.cad_graph_converter import create as create_graph


class MsvNetDataSet(InMemoryDataset):
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

    def process(self):
        path = self.raw_data_root

        for root, dirs, files in os.walk(self.raw_data_root):

            for file in files:

                file_labels = []

                if file.lower().endswith('.csv'):
                    with open(f'{root}/{file}', 'r', encoding='utf-8') as f:

                        for line in f.readlines():
                            file_labels.append(int(line.split(",")[-1]))

                        file_name = str(file).replace('.csv', '.STL')
                        self.data_list.append(create_graph(root + '/' + file_name, file_labels))
                        print(file_name)

        torch.save(self.collate(self.data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
