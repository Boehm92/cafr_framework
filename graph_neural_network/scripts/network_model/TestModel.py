import torch
from torch.nn import Linear
import torch.nn.functional as f
from sklearn.metrics import f1_score
from torch_geometric.nn import GraphConv as GraphConvLayer # FeaStConv, GATConv, GraphConv GravNetConv, ,GCNConv, PointGNNConv
from torch_geometric.nn import global_mean_pool

class TestModel(torch.nn.Module):
    def __init__(self, dataset, device, batch_size, dropout_probability, number_conv_layers, hidden_channels):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.dropout_probability = dropout_probability
        self.number_conv_layers = number_conv_layers
        self.conv_layers = []

        self.conv1 = GraphConvLayer(dataset.num_node_features, int(hidden_channels), aggr='mean')
        self.conv_layers.append(self.conv1)
        if self.number_conv_layers > 1:
            self.conv2 = GraphConvLayer(int(hidden_channels), int(hidden_channels), aggr='mean')
            self.conv_layers.append(self.conv2)
        if self.number_conv_layers > 2:
            self.conv3 = GraphConvLayer(int(hidden_channels), int(hidden_channels), aggr='mean')
            self.conv_layers.append(self.conv3)
        if self.number_conv_layers > 3:
            self.conv4 = GraphConvLayer(int(hidden_channels), int(hidden_channels), aggr='mean')
            self.conv_layers.append(self.conv4)
        if self.number_conv_layers > 4:
            self.conv5 = GraphConvLayer(int(hidden_channels), int(hidden_channels), aggr='mean')
            self.conv_layers.append(self.conv5)
        if self.number_conv_layers > 5:
            self.conv6 = GraphConvLayer(int(hidden_channels), int(hidden_channels), aggr='mean')
            self.conv_layers.append(self.conv6)
        if self.number_conv_layers > 6:
            self.conv7 = GraphConvLayer(int(hidden_channels), int(hidden_channels), aggr='mean')
            self.conv_layers.append(self.conv7)

        self.lin_out = Linear(int(hidden_channels), 24)

    def forward(self, x, edge_index, batch):
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = f.dropout(x, p=self.dropout_probability, training=self.training)
        x = self.lin_out(x)

        return x

    def train_model(self, loader, criterion, optimizer):
        self.train()

        total_loss = 0
        for i, data in enumerate(loader):  # Iterate in batches over the training dataset.
            data = data.to(self.device)
            optimizer.zero_grad()  # Clear gradients.
            out = self(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y.reshape(self.batch_size, -1))
            total_loss += loss.item() * data.num_graphs
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

        return total_loss / len(loader.dataset)

    def val_loss(self, loader, criterion):
        self.eval()

        total_loss = 0
        for data in loader:  # Iterate in batches over the training dataset.
            data = data.to(self.device)
            out = self(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y.reshape(self.batch_size, -1))
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def val_model(self, loader):
        self.eval()

        ys, preds = [], []
        for data in loader:
            ys.append(data.y.reshape(self.batch_size, -1))
            out = self(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device))
            preds.append((out > 0).float().cpu())

        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

    @torch.no_grad()
    def test_model(self, loader):
        self.eval()

        ys, preds = [], []
        for data in loader:
            ys.append(data.y.reshape(self.batch_size, -1))
            out = self(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device))
            preds.append((out > 0).float().cpu())
        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

        return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

