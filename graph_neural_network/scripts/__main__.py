import os
import torch
import optuna
import argparse
from time import time
from data_importer.GraphImporter import GraphImporter
from MachiningFeatureRecognizer import MachiningFeatureRecognizer

_device = "cuda" if torch.cuda.is_available() else "cpu"
_test_dataset = GraphImporter(os.getenv('TEST_DATASET_SOURCE'), os.getenv('TEST_DATASET_DESTINATION'))
_training_dataset = GraphImporter(
    os.getenv('TRAINING_DATASET_SOURCE'), os.getenv('TRAINING_DATASET_DESTINATION')).shuffle()

_parser = argparse.ArgumentParser(description='Base configuration of the synthetic data generator')
_parser.add_argument('--application_mode',
                     dest='application_mode', default='test', type=str,
                     help='can be training or test')
_parser.add_argument('--study_name',
                     dest='study_name', default='CAFR_Generalization_Experiment', type=str,
                     help='')
_parser.add_argument('--project_name',
                     dest='project_name', default='test_project', type=str,
                     help='')
_parser.add_argument('--device',
                     dest='device', default=_device, type=str,
                     help='')
_parser.add_argument('--train_val_partition',
                     dest='train_val_partition', default=300, type=int,
                     help='')
_parser.add_argument('--training_dataset',
                     dest='training_dataset', default=_training_dataset,
                     help='')
_parser.add_argument('--test_dataset',
                     dest='test_dataset', default=_test_dataset,
                     help='')
_parser.add_argument('--max_epoch',
                     dest='max_epoch', default=2, type=int,
                     help='100')
_parser.add_argument('--max_network_layer',
                     dest='max_network_layer', default=7, type=int,
                     help='')
_parser.add_argument('--h_channels',
                     dest='h_channels', default=512,
                     help='[32, 64, 128, 256, 512]')
_parser.add_argument('--batch_size',
                     dest='batch_size', default=1,
                     help='[32, 64, 128, 256]')
_parser.add_argument('--learning_rate',
                     dest='learning_rate', default=0.001,
                     help='[0.01, 0.001, 0.0001]')
_parser.add_argument('--dropout_probability',
                     dest='dropout_probability', default=0.3, type=float,
                     help='0.5')
_parser.add_argument('--schedular_gamma',
                     dest='schedular_gamma', default=0.001, type=float,
                     help='')
_parser.add_argument('--schedular_step_size',
                     dest='schedular_step_size', default=30, type=int,
                     help='')

if __name__ == '__main__':
    _config = _parser.parse_args()
    _machining_feature_recognizer = MachiningFeatureRecognizer(_config)

    if _config.application_mode == "training":
        _study = optuna.create_study(direction="maximize", study_name=_config.study_name)
        _study.optimize(lambda trial: _machining_feature_recognizer.training(trial), n_trials=1)

    elif _config.application_mode == "test":
        _start_time = time()
        response = _machining_feature_recognizer.test()
        _end_time = time()

        print("Inference time: ", (_end_time - _start_time))
        print("Test Accuracy: ", response)
        print(f"For {len(_test_dataset)} models")
