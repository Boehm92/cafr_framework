import torch


def train(model, loader, criterion, optimizer):
    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_loss = 0

    for data in loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.

        loss = criterion(out, data.y.reshape(16, -1))
        total_loss += loss.item() # * data.num_graphs
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

    return total_loss / len(loader.dataset)


def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.
