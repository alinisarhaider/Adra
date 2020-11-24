import logging
import os
from pathlib import Path

import absl.logging
import yaml

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ----------------------------------------------------
# Global Parameters for experiment
# ----------------------------------------------------
if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read global variables from YAML file
configurations_path = 'config.yaml'
with open(configurations_path, 'r') as stream:
    config = yaml.safe_load(stream)['parameters']


GLOBAL_VARS = {
    'TRAINING_MODELS': config['artifacts']['TRAINING_MODELS'],
    'BEST_BY_VALIDATION_MODELS': config['artifacts']['BEST_BY_VALIDATION_MODELS'],
    'LAST_EPOCH_MODELS': config['artifacts']['LAST_EPOCH_MODELS'],
    'TRAINED_MODEL': config['artifacts']['FINALIZED_MODELS'],
    'PREDICTED_LABELS': config['artifacts']['PREDICTED_LABELS'],
    'DATASET_ROOT': config['dataset_path']['DATASET_ROOT'],
    'CLASS1_PATH': config['dataset_path']['CLASS1_PATH'],
    'CLASS2_PATH': config['dataset_path']['CLASS2_PATH'],
    'TEST_PATH': config['dataset_path']['TEST_PATH'],
    'FLIP_UPSIDE_DOWN': config['dataset_augmentation']['FLIP_UPSIDE_DOWN'],
    'FLIP_LEFT_TO_RIGHT': config['dataset_augmentation']['FLIP_LEFT_TO_RIGHT'],
    'AUGMENT_TRAIN': config['dataset_augmentation']['AUGMENT_TRAIN'],
    'AUGMENT_VAL': config['dataset_augmentation']['AUGMENT_VAL'],
    'AUGMENT_TEST': config['dataset_augmentation']['AUGMENT_TEST'],
    'SEED': config['dataset_split']['SEED'],
    'NORMALIZE_PIXEL_VALUES': config['dataset_normalization']['NORMALIZE_PIXEL_VALUES']
}

Path(GLOBAL_VARS['TRAINING_MODELS']).mkdir(parents=True, exist_ok=True)
Path(GLOBAL_VARS['BEST_BY_VALIDATION_MODELS']).mkdir(parents=True, exist_ok=True)
Path(GLOBAL_VARS['LAST_EPOCH_MODELS']).mkdir(parents=True, exist_ok=True)
Path(GLOBAL_VARS['PREDICTED_LABELS']).mkdir(parents=True, exist_ok=True)

# %% Main Function
logging.info('Main Function...')
