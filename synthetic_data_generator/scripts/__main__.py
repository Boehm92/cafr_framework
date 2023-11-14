import argparse
from utils.DataGenerator import DataGenerator

_parser = argparse.ArgumentParser(description='Base configuration of the synthetic data generator')
_parser.add_argument('--target_directory',
                     dest='target_directory', default='TRAINING_DATASET_SOURCE', type=str,
                     help='The variables TRAINING_DATASET_SOURCE and TEST_DATASET_SOURCE are environment variables used'
                          'to access the training and test cad data in the CAFR framework. Please exchange the '
                          'variables if you want to generate training or test data')

_parser.add_argument('--max_machining_feature_dimension',
                     dest='max_machining_feature_dimension', default=0.6, type=float,
                     help='This value limits the length/with or radius of every machining feature in the'
                          'MachiningFeature class. The range of this value should be 0.1-1')

_parser.add_argument('--cad_data_generation_start_cycle',
                     dest='cad_data_generation_start_cycle', type=int, default=1,
                     help='This value defines with wich ID the data generation process starts. This can be important'
                          'if a dataset was already created and has to be increased with additional data. If the'
                          'start number of the data generation cycle then is not adapted, the existing data is just'
                          'overwritten')

_parser.add_argument('--cad_data_generation_cycles',
                     dest='cad_data_generation_cycles', type=int, default=14746,
                     help='This value defines how many cad models with multiple machining feature are '
                          'created.')

_parser.add_argument('--max_machining_feature_count',
                     dest='max_machining_feature_count', type=int, default=3,
                     help='This value defines how many machining feature maximal can be applied to the base'
                          'primitive. The actual value will be randomly chosen from an interval from 1 to'
                          'the here defined value. For single feature creation this value should be 2, because the '
                          'random function starts at 1. If machining feature should not be selected randomly, the value'
                          ' machining_feature_id in the Class DataGenerator has to be fixed to the wished machining '
                          'feature')

_parser.add_argument('--rotate_base_primitive',
                     dest='rotate_base_primitive', type=bool, default=True,
                     help='When true the cad model, after the application of each machining feature, will'
                          'be rotated around x, y or z randomly with -90 or 90 degrees also randomly '
                          'selected')

_parser.add_argument('--select_machining_feature_id_random',
                     dest='select_machining_feature_id_random', type=bool, default=True,
                     help='When true, the machining_feature_id in the Class DataGenerator is selected '
                          'randomly. If false it uses the config value machining_feature_id. Should only'
                          'be true for single machining feature creation.')

_parser.add_argument('--machining_feature_id',
                     dest='machining_feature_id', type=int, default=0,
                     help='This value selects a specific machining feature for the application to the base'
                          'primitive. Should only be true for single machining feature creation.')

if __name__ == '__main__':
    _config = _parser.parse_args()
    _data_generator = DataGenerator(_config)
    _data_generator.generate()
