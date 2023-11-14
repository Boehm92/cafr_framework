import torch
from torch.nn import Linear
import torch.nn.functional as f
from sklearn.metrics import f1_score
from torch_geometric.nn import GraphConv as GraphConvLayer
from torch_geometric.nn import global_mean_pool


class TestingNetwork(torch.nn.Module):
    def __init__(self, dataset, device, batch_size=1, dropout_probability=0.3, hidden_channels=512):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.dropout_probability = dropout_probability
        self.hidden_channels = hidden_channels

        self.conv1 = GraphConvLayer(self.dataset.num_node_features, int(self.hidden_channels))
        self.conv2 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        self.conv3 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        self.conv4 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        self.conv5 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        self.conv6 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        self.conv7 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        self.lin_out = Linear(int(self.hidden_channels), 24)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)
        x = x.relu()
        x = f.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.conv6(x, edge_index)
        x = x.relu()
        x = self.conv7(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = f.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin_out(x)

        return x

    @torch.no_grad()
    def val_model(self, loader):
        self.eval()

        label_list, prediction_list = [], []
        for data in loader:
            label_list.append(data.y.reshape(-1, 24))  # for some unknown reason, in the testing val_model function, we
            # need to write the class number instead of the batch size.
            out = self(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device))
            prediction_list.append((out > 0).float().cpu())

        label, prediction = torch.cat(label_list, dim=0).numpy(), torch.cat(prediction_list, dim=0).numpy()

        return (100 * (f1_score(label, prediction, average='micro') if prediction.sum() > 0 else 0)), label, prediction
