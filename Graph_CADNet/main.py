import torch
from GraphNeuralNetwork.ModelMethods import inference, test, train
from Graph_CADNet.DataImporter.StlDataset import StlDataset
from torch_geometric.loader import DataLoader
from GraphNeuralNetwork.Model import GCN

training_dataset = StlDataset('../Data/CadData/Training', '../Data/GraphData/Training')
torch.manual_seed(12345)
training_dataset = training_dataset.shuffle()

train_dataset = training_dataset[:2000]
test_dataset = training_dataset[2000:]
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

inference_dataset = StlDataset('../Data/CadData/Inference', '../Data/GraphData/Inference')
inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

model = GCN(hidden_channels=64, dataset=training_dataset)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

training_mode = True

if training_mode:
    for epoch in range(1, 10):
        train(model, train_loader, criterion, optimizer)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:03d} ,Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    torch.save(model.state_dict(), 'GraphNeuralNetwork/Weights.pt')
else:
    inference(model, inference_loader, 'GraphNeuralNetwork/Weights.pt')
