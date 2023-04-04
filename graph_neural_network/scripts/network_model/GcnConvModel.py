import torch
from torch.nn import Linear
import torch.nn.functional as f
from sklearn.metrics import f1_score
# bevor next run change convlayer count!!!
from torch_geometric.nn import GraphConv as GraphConvLayer
from torch_geometric.nn import global_mean_pool

class GraphConvModel(torch.nn.Module):
    def __init__(self, dataset, device, batch_size=64, dropout_probability=0.4, hidden_channels=256):
        super().__init__()
        self.device = device
        self.device = device
        self.batch_size = batch_size
        self.dropout_probability = dropout_probability

        self.conv1 = GraphConvLayer(dataset.num_node_features, int(hidden_channels))
        self.conv2 = GraphConvLayer(int(hidden_channels), int(hidden_channels))
        self.conv3 = GraphConvLayer(int(hidden_channels), int(hidden_channels))
        self.conv4 = GraphConvLayer(int(hidden_channels), int(hidden_channels))
        self.conv5 = GraphConvLayer(int(hidden_channels), int(hidden_channels))
        self.conv6 = GraphConvLayer(int(hidden_channels), int(hidden_channels))
        self.conv7 = GraphConvLayer(int(hidden_channels), int(hidden_channels))
        self.lin_out = Linear(int(hidden_channels), 24)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = f.relu(self.conv1(x, edge_index))
        x = f.relu(self.conv2(x, edge_index))
        x = f.relu(self.conv3(x, edge_index))
        x = f.relu(self.conv4(x, edge_index))
        x = f.relu(self.conv5(x, edge_index))
        x = f.relu(self.conv6(x, edge_index))
        x = f.relu(self.conv7(x, edge_index))

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
    def test_model(self, loader, batch):
        self.eval()

        ys, preds = [], []
        for data in loader:
            ys.append(data.y.reshape(batch, -1))
            out = self(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device))
            preds.append((out > 0).float().cpu())
        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

        return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

