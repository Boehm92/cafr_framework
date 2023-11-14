import os
import torch
import wandb
import optuna
from torch_geometric.loader import DataLoader
from network_model.TestModel import TestModel
from network_model.GraphConvModel import GraphConvModel
import torch.optim.lr_scheduler


class MachiningFeatureRecognizer:
    def __init__(self, _config):
        self.max_epoch = _config.max_epoch
        self.max_network_layer = _config.max_network_layer
        self.hidden_channels = _config.h_channels
        self.batch_size = _config.batch_size
        self.learning_rate = _config.learning_rate
        self.dropout_probability = _config.dropout_probability
        self.schedular_step_size = _config.schedular_step_size
        self.schedular_gamma = _config.schedular_gamma
        self.training_dataset = _config.training_dataset
        self.test_dataset = _config.test_dataset
        self.train_val_partition = _config.train_val_partition
        self.study_name = _config.study_name
        self.project_name = _config.project_name
        self.device = _config.device
        torch.manual_seed(1)

    def training(self, trial):
        # Hyperparameter optimization
        _best_accuracy = 0
        _batch_size = trial.suggest_categorical("batch_size", self.batch_size)
        _dropout_probability = trial.suggest_float("dropout_probability", 0.1, self.dropout_probability, step=0.1)
        _number_conv_layers = trial.suggest_int("number_conv_layers", 2, self.max_network_layer)
        _hidden_channels = trial.suggest_categorical("hidden_channel", self.hidden_channels)
        _learning_rate = trial.suggest_categorical("learning_rate", self.learning_rate)

        # Configuring graph neural network
        # Please note that following configuration can be used for the GraphConv and GCNConv class from pytorch
        # geometric. You simply have to exchange TestModel class the imported GraphConvLayer. If you want to use the
        # FeaStConv layer for the hyperparameter optimization and training, you need to define an attention_head
        # variable in the above Hyperparameter optimization section and also configure the GraphConvModel class
        # accordingly. For more information of the FeaStConv layer, please go to
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.FeaStConv.html#torch_geometric.nn.conv.FeaStConv
        # If you want to test another GNNLayer than GraphConv, you need to define a new class after training procedure
        # following the structure of the GraphConvModel class in this framework. Then simply exchange the GraphConvModel
        # class in this class at the method test with your own new GNN Model class.

        self.training_dataset.shuffle()
        train_loader = DataLoader(self.training_dataset[:self.train_val_partition], batch_size=_batch_size,
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(self.training_dataset[self.train_val_partition:], batch_size=_batch_size, shuffle=True,
                                drop_last=True)
        model = TestModel(dataset=self.training_dataset, device=self.device, batch_size=_batch_size,
                          dropout_probability=_dropout_probability, number_conv_layers=_number_conv_layers,
                          hidden_channels=_hidden_channels).to(self.device)
        print("b_size: ", _batch_size)
        print("lr: ", _learning_rate)
        print("dropout_probability: ", _dropout_probability)
        print("Graph neural network: ", model)

        # Configuring learning functions
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=_learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.schedular_step_size,
                                                    gamma=self.schedular_gamma, verbose=True)

        # Setting up hyperparameter function and wandb
        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb.init(project=self.project_name, entity="boehm92", config=config, group=self.study_name, reinit=True)

        # Training
        for epoch in range(1, self.max_epoch):
            training_loss = model.train_model(train_loader, criterion, optimizer)
            val_loss = model.val_loss(val_loader, criterion)
            train_f1 = model.val_model(train_loader)
            val_f1 = model.val_model(val_loader)
            trial.report(val_f1, epoch)
            scheduler.step()

            wandb.log({'training_loss': training_loss, 'val_los': val_loss, 'train_F1': train_f1, 'val_F1': val_f1})

            if (best_accuracy < val_f1) & ((val_loss - training_loss) < 0.04):
                torch.save(model.state_dict(), os.getenv('WEIGHTS') + '/weights.pt')
                best_accuracy = val_f1
                print("Saved model due to better found accuracy")

            if trial.should_prune():
                wandb.run.summary["state"] = "pruned"
                wandb.finish(quiet=True)
                raise optuna.exceptions.TrialPruned()

            print(f'Epoch: {epoch:03d}, training_loss: {training_loss:.4f}, val_los: {val_loss:.4f}, '
                  f'train_F1: {train_f1:.4f}, val_F1: {val_f1:.4f}')

        wandb.run.summary["Final F-Score"] = val_f1
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)

        return val_f1

    def test(self):
        # Configuring graph neural network
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        model = GraphConvModel(dataset=self.test_dataset, device=self.device).to(self.device)
        model.load_state_dict(torch.load(os.getenv('WEIGHTS') + '/weights.pt', map_location=self.device))

        print("b_size: ", self.batch_size)
        print("lr: ", self.learning_rate)
        print("dropout_probability: ", self.dropout_probability)
        print("Graph neural network: ", model)

        test_f1 = model.val_model(test_loader)

        return test_f1
