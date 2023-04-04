import torch
from torch.nn import Linear
import torch.nn.functional as f
from sklearn.metrics import f1_score
from torch_geometric.nn import FeaStConv as GraphConvLayer
from torch_geometric.nn import global_mean_pool


class FeastModel(torch.nn.Module):
    def __init__(self, dataset, device, batch_size=32, dropout_prob=0.2,
                 attention_heads=8, hidden_channels=16):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.attention_heads = attention_heads
        self.hidden_channels = hidden_channels

        self.fc0 = Linear(dataset.num_node_features, hidden_channels)
        self.conv1 = GraphConvLayer(hidden_channels, hidden_channels * 2, heads=attention_heads)
        self.conv2 = GraphConvLayer(hidden_channels * 2, hidden_channels * 4, heads=attention_heads)
        self.conv3 = GraphConvLayer(hidden_channels * 4, hidden_channels * 8, heads=attention_heads)
        self.lin_out = Linear(hidden_channels * 8, 24)

    def forward(self, x, edge_index, batch):
        x = f.elu(self.fc0(x))
        x = f.elu(self.conv1(x, edge_index))
        x = f.elu(self.conv2(x, edge_index))
        x = f.elu(self.conv3(x, edge_index))

        # Readout layer
        x = global_mean_pool(x, batch)
        x = f.dropout(x, p=self.dropout_prob, training=self.training)

        # Final classifier
        x = self.lin_out(x)
        return x

    def train_model(self, loader, criterion, optimizer):
        self.train()
        total_loss = 0
        for i, data in enumerate(loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            out = self(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.reshape(self.batch_size, -1))
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        return total_loss / len(loader.dataset)

    def val_loss(self, loader, criterion):
        self.eval()
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self(data.x, data.edge_index, data.batch)
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
