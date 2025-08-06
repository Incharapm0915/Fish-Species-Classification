"""
Configuration file for Fish Classification Project
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Dataset paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VAL_DIR = os.path.join(DATA_DIR, 'val')

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32

# Training parameters
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Data augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
FILL_MODE = 'nearest'

# Pre-trained models to experiment with
PRETRAINED_MODELS = [
    'VGG16',
    'ResNet50',
    'MobileNet',
    'InceptionV3',
    'EfficientNetB0'
]

# Model saving parameters
MODEL_SAVE_FORMAT = '.h5'
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.h5')

# Results and logging
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')

# Create directories if they don't exist
for directory in [MODELS_DIR, RESULTS_DIR, LOGS_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)