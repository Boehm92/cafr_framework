import torch
from torch.nn import Linear
import torch.nn.functional as f
from sklearn.metrics import f1_score
from torch_geometric.nn import GraphConv as GraphConvLayer  # FeaStConv, GCNConv
from torch_geometric.nn import global_mean_pool


class TrainingNetwork(torch.nn.Module):
    def __init__(self, dataset, device, batch_size, dropout_probability, number_conv_layers, hidden_channels):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.dataset = dataset
        self.hidden_channels = hidden_channels
        self.dropout_probability = dropout_probability
        self.number_conv_layers = number_conv_layers

        self.conv1 = GraphConvLayer(self.dataset.num_node_features, int(self.hidden_channels))
        if self.number_conv_layers > 1:
            self.conv2 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        if self.number_conv_layers > 2:
            self.conv3 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        if self.number_conv_layers > 3:
            self.conv4 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        if self.number_conv_layers > 4:
            self.conv5 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        if self.number_conv_layers > 5:
            self.conv6 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')
        if self.number_conv_layers > 6:
            self.conv7 = GraphConvLayer(int(self.hidden_channels), int(self.hidden_channels), aggr='mean')

        self.lin_out = Linear(int(self.hidden_channels), 24)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        if self.number_conv_layers > 1:
            x = self.conv2(x, edge_index)
            x = x.relu()
        if self.number_conv_layers > 2:
            x = self.conv3(x, edge_index)
            x = x.relu()
        if self.number_conv_layers > 3:
            x = self.conv4(x, edge_index)
            x = x.relu()
        if self.number_conv_layers > 4:
            x = self.conv5(x, edge_index)
            x = x.relu()
        if self.number_conv_layers > 5:
            x = self.conv6(x, edge_index)
            x = x.relu()
        if self.number_conv_layers > 6:
            x = self.conv7(x, edge_index)
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

        label_list, prediction_list = [], []
        for data in loader:
            label_list.append(data.y.reshape(self.batch_size, -1))
            out = self(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device))
            prediction_list.append((out > 0).float().cpu())

        label, prediction = torch.cat(label_list, dim=0).numpy(), torch.cat(prediction_list, dim=0).numpy()
        return 100 * (f1_score(label, prediction, average='micro') if prediction.sum() > 0 else 0)
