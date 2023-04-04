import os
import torch
import wandb
import optuna
from data_importer.MsvNetDataSet import MsvNetDataSet
from torch_geometric.loader import DataLoader
from network_model.GraphConvModel import GraphConvModel
from network_model.TestModel import TestModel

torch.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
data_partition = 98304

test_dataset = MsvNetDataSet(os.getenv('TEST_DATASET_SOURCE'), os.getenv('TEST_DATASET_DESTINATION')).shuffle()
# training_dataset = MsvNetDataSet(os.getenv('TRAINING_DATASET_SOURCE'),
#                                  os.getenv('TRAINING_DATASET_DESTINATION')).shuffle()

STUDY_NAME = "MRF_GNN"


def objective(trial):
    # Hyperparameter
    max_epoch = 2
    # number_conv_layers = 6 trial.suggest_int("number_conv_layers", 2, 7)
    # h_channel = 512  # trial.suggest_categorical("h_channel", [16, 32, 64, 128, 256, 512])
    # b_size = 32  # trial.suggest_categorical("b_size", [16, 32, 64, 128, 256])
    # lr = 0.001  # trial.suggest_categorical("lr", [0.02, 0.01, 0.002, 0.001, 0.0002, 0.0001])
    # dropout_probability = 0.2  # trial.suggest_float("dropout_probability", 0.1, 0.5, step=0.1)

    number_conv_layers = trial.suggest_int("number_conv_layers", 6, 6)
    h_channel = trial.suggest_categorical("h_channel", [512])
    b_size = trial.suggest_categorical("b_size", [1]) #16
    lr = trial.suggest_categorical("lr", [0.001])
    dropout_probability = trial.suggest_float("dropout_probability", 0.2, 0.2)

    print("b_size: ", b_size)
    print("lr: ", lr)
    print("dropout_probability: ", dropout_probability)

    # training_dataset.shuffle()
    test_dataset.shuffle()
    # train_loader = DataLoader(training_dataset[:data_partition], batch_size=b_size, shuffle=True, drop_last=True)
    # val_loader = DataLoader(training_dataset[data_partition:], batch_size=b_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False, drop_last=True)

    # model = GraphConvModel(dataset=training_dataset, device=device, batch_size=b_size,
    #                        dropout_probability=dropout_probability, hidden_channels=h_channel).to(device)

    model = GraphConvModel(dataset=test_dataset, device=device, batch_size=b_size,
                           dropout_probability=dropout_probability, hidden_channels=h_channel).to(device)
    model.load_state_dict(torch.load(os.getenv('WEIGHTS') + '/weights.pt'))
    print(model)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    config = dict(trial.params)
    config["trial.number"] = trial.number
    wandb.init(project="TEST_RUN",
               entity="boehm92",
               config=config,
               group=STUDY_NAME,
               reinit=True,
               )

    for epoch in range(1, max_epoch):
        # training_loss = model.train_model(train_loader, criterion, optimizer)
        # val_loss = model.val_loss(val_loader, criterion)
        # train_f1 = model.val_model(train_loader)
        # val_f1 = model.val_model(val_loader)
        test_f1 = model.test_model(test_loader, b_size)
        trial.report(test_f1, epoch)

        # wandb.log({'training_loss': training_loss, 'val_los': val_loss, 'train_F1': train_f1,
        #            'val_F1': val_f1})  # ,'test_F1': test_f1
        wandb.log({'test_F1': test_f1})

        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()

        # print(f'Epoch: {epoch:03d}, training_loss: {training_loss:.4f}, val_los: {val_loss:.4f}, '
        #       f'train_F1: {train_f1:.4f}, val_F1: {val_f1:.4f}')

        print(f'Epoch: {epoch:03d}, test_F1: {test_f1:.4f}')

    # report the final validation accuracy to wandb
    wandb.run.summary["Final F-Score"] = test_f1  # val_f1
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    # torch.save(model.state_dict(), os.getenv('WEIGHTS') + '/weights.pt')

    # return val_f1
    return test_f1


if __name__ == '__main__':
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
    )
    study.optimize(objective, n_trials=1)
