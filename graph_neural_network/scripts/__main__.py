import os
import torch
import optuna
import argparse
from time import time
from data_importer.GraphImporter import GraphImporter
from MachiningFeatureRecognizer import MachiningFeatureRecognizer

_parser = argparse.ArgumentParser(description='Base configuration of the synthetic data generator')

# Basic configuration of the cafr framework
_parser.add_argument('--application_mode',
                     dest='application_mode', default='training', type=str,
                     help='The application modes has "trained" and "test". When set to trained the framework uses the'
                          'TestModel class to train graph neural network. Please note, if you want to test different'
                          'graph conv layer, the TestModel class must be configured with accordingly. For example,'
                          'if you want to use the FeastNet layer please follow the guidelines from '
                          'https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv - '
                          '.FeaStConv.html#torch_geometric.nn.conv.FeaStConv, When set to test, the GraphConvModel is'
                          'used. Here a weights file for a trained model is loaded. When you trained the model first,'
                          'the weights file should be automatically be saved in the given path from the defined'
                          'environment variable WEIGHTS, like described in the README file. Do to the fact, that the'
                          'trained mode utilizes hyperparameter, the model architecture can vary with every program run'
                          'Therefore, make sure that you configure the GraphConvModel accordingly. Also the training '
                          'procedure uses a so called hyperparameter optimization. For more info about this '
                          'optimization process, please visit: https://optuna.org/'
                     )
_parser.add_argument('--hyperparameter_trials',
                     dest='hyperparameter_trials', default=100, type=int,
                     help='The hyperparameter_trials value defines how often the training procedure is repeated.'
                          'Reason for repeated training is, that this framework applies a hyperparameter optimization '
                          'for the training procedure. Here, an optimization algorithm tries to find the best hyper '
                          'parameters like hidden channels or amount of graph conv layers. To find the best hyper '
                          'parameter, the training procedure has to be conducted multiple times, so the optimization '
                          'algorithms can analyze how different values of each parameter influences the training '
                          'procedure of the network. More information about hyperparameter optimization at:'
                          'https://optuna.org/'
                     )
_parser.add_argument('--project_name',
                     dest='project_name', default='machining_feature_recognition', type=str,
                     help='This name belongs to the wandb project which is created when the code is started. The wandb '
                          'code publishes training parameters like train_accuracy to your personal wandb dashboard. You'
                          'just have to register at www.wandb.ai and follow the instructions at: '
                          'https://docs.wandb.ai/quickstart.'
                          '(Not used for testing)')
_parser.add_argument('--study_name',
                     dest='study_name', default='test_1', type=str,
                     help='The study name defines a subgroup for the wandb project, which is defined above. This helps'
                          'to repeat an experiment or training process without creating every time a new wandb project.'
                          '(Not used for testing)')
_parser.add_argument('--device',
                     dest='device', default=("cuda" if torch.cuda.is_available() else "cpu"), type=str,
                     help='The device variable defines if the code is run on the gpu or the cpu. If you installed the'
                          'cuda toolkit and the related pytorch and pytorch-geometric packages, then the code should '
                          'run on the gpu. If not, the code automatically will run on the cpu, which will be far '
                          'slower. Please note, that in the requirements1 and 2 files, there are example of '
                          'installations setting, however the necessary python packages vary strongly in regard to'
                          'used operation system, python interpreter, used graphic card and installed cuda toolkit.'
                          'So, it may take some time to find the right setting for you. We suggest, for the first '
                          'implementation, to install the packages manually.')
_parser.add_argument('--train_val_partition',
                     dest='train_val_partition', default=22000, type=int,
                     help='This variable allows you to separate the training data, taken from the "data -> cad ->'
                          'training" folder, into training and validation datasets. For example, if you have 24000 '
                          'cad models, if you type in value 22000 models, then 22000 models will be utilized for '
                          'training and 2000 models for validation. NOTE: The cad models will be first converted into a'
                          'fitting graph representation and saved into the data -> graph -> training folder.'
                          '(Not used for testing)')
_parser.add_argument('--training_dataset',
                     dest='training_dataset', default=GraphImporter(os.getenv('TRAINING_DATASET_SOURCE'),
                                                                    os.getenv('TRAINING_DATASET_DESTINATION')).shuffle()
                     , help='The training_dataset config holds the training data. The data is converted via the'
                            'GraphImporter class into a fitting graph representation and then loaded for training. The'
                            'conversion process takes some time, especially for larger data, but it has to be done only'
                            'once, as long as the data doesnt change. '
                            '(Not used for testing)')
_parser.add_argument('--test_dataset',
                     dest='test_dataset', default=GraphImporter(os.getenv('TEST_DATASET_SOURCE'),
                                                                os.getenv('TEST_DATASET_DESTINATION')),
                     help='The test_dataset config holds the test data. The process is the same as for the '
                          'training_data. NOTE that the batch size should be must be configured accordingly to the size'
                          ' of the test_dataset. Larger batch_size means also faster run time.')
_parser.add_argument('--max_epoch',
                     dest='max_epoch', default=100, type=int,
                     help='The max epoch defines how often the complete training data is run trough. One epoch means'
                          'therefore, that the graph neural network is fitted ones an all available training data. '
                          'More epochs generally decreases the network loss, but can also lead to overfitting,'
                          'memorizing the training data and not be applicable to new data. We utilized 100 epochs for'
                          'most of our experiments')

# Graph neural network configurations
_parser.add_argument('--max_network_layer',
                     dest='max_network_layer', default=[1, 2, 3, 4, 5, 6, 7], type=int,
                     help='The max_network_layer defines how many graph layers the model has. For training mode, please'
                          'use following array [1, 2, 3, 4, 5, 6, 7], which the hyper-optimization can choose from'
                          'accordingly. For inference mode, you must give an array with only one element [7], the'
                          'number of the best found layer config during the training and hyperparameter-optimization'
                          'For more information how graph neural networks and their layers work please visit:'
                          'https://pytorch-geometric.readthedocs.io/en/latest/index.html')
_parser.add_argument('--h_channels',
                     dest='h_channels', default=[32, 64, 128, 256, 512],
                     help='The hidden channel config defines how many feature vectors or gathered information from each'
                          'layer is passed to the next. For training and test the procedure is the same as for the'
                          'network_layer. For training we suggest an array with following elements:'
                          '[32, 64, 128, 256, 512], For testing, choose the best value again, for example [512]')
_parser.add_argument('--batch_size',
                     dest='batch_size', default=[32, 64, 128, 256],
                     help='The batch size of the graph neural network defines how many cad models for one training step'
                          '(training step is not equal to epoch) are used. The higher the batch size the faster the'
                          'training procedure, but this also can make the training worse. Lower batch sizes can help'
                          'the training procedure but also can lead to overfitting. For the training procedure we'
                          'suggest an array with following elements [32, 64, 128, 256]. For testing we suggest to use'
                          'a batch size of [1]')
_parser.add_argument('--learning_rate',
                     dest='learning_rate', default=[0.01, 0.001, 0.0001],
                     help='The learning rate defines the speed of the training process. Higher learning learning rate'
                          'can lead to faster, but maybe also to unstable training. Lower learning rate can lead'
                          'to a more staple but also slower training. Also, a longer training procedure due to a lower'
                          'learning rate can lead to overfitting. For our experiments, a learning rate of [0.001]'
                          'showed good results. '
                          '(Not used for testing)')
_parser.add_argument('--dropout_probability',
                     dest='dropout_probability', default=[0.1, 0.2, 0.3, 0.4, 0.5], type=float,
                     help='The dropout_probability for the graph neural network disables randomly some neurons. The '
                          'purpose is to decrease overfitting. Higher values, are necessary when the model starts early'
                          'to over fit (memorizing training data), but also can lead to unstable training.'
                          'For training you should use [0.1, 0.2, 0.3, 0.4, 0.5]'
                          '(Not used for testing)')
_parser.add_argument('--schedular_gamma',
                     dest='schedular_gamma', default=0.001, type=float,
                     help='The schedular gamma is a factor to decrease the learning rate after a certain time. The'
                          'value 0.001 achieved good results for us'
                          '(Not used for testing)')
_parser.add_argument('--schedular_step_size',
                     dest='schedular_step_size', default=30, type=int,
                     help='The schedular step size defines when the learning rate should be decrease during training. '
                          'For the decrease, the current learning rate is multiplied with the schedular_gamma. The'
                          'value 30 achieved good results for us.'
                          '(Not used for testing)')

if __name__ == '__main__':
    _config = _parser.parse_args()
    _machining_feature_recognizer = MachiningFeatureRecognizer(_config)

    if _config.application_mode == "training":

        print("")
        print("The training is conducted on: ", _config.device)

        _study = optuna.create_study(direction="maximize", study_name=_config.study_name)
        _study.optimize(lambda trial: _machining_feature_recognizer.training(trial),
                        n_trials=_config.hyperparameter_trials)
    elif _config.application_mode == "test":
        _start_time = time()
        response, label, prediction = _machining_feature_recognizer.test()
        _end_time = time()

        print("")
        print("The testing is conducted on: ", _config.device)
        print("Run time: ", (_end_time - _start_time))
        print("Test Accuracy: ", response)
        print(f"For {len(_config.test_dataset)} models")
        print("")
        print("Label and Network Output comparison: ")
        print("Label-List: ", label)
        print("Network-Prediction: ", prediction)
