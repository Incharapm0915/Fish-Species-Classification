# Preprocessing Configuration for Fish Classification
# Auto-generated configuration file

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32
RESCALE = 0.00392156862745098

# Dataset info
NUM_CLASSES = 11
CLASS_NAMES = ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel', 'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet', 'fish sea_food trout']
CLASS_INDICES = {'animal fish': 0, 'animal fish bass': 1, 'fish sea_food black_sea_sprat': 2, 'fish sea_food gilt_head_bream': 3, 'fish sea_food hourse_mackerel': 4, 'fish sea_food red_mullet': 5, 'fish sea_food red_sea_bream': 6, 'fish sea_food sea_bass': 7, 'fish sea_food shrimp': 8, 'fish sea_food striped_red_mullet': 9, 'fish sea_food trout': 10}

# Augmentation strategy
AUGMENTATION_LEVEL = "MODERATE"
AUGMENTATION_PARAMS = {'rescale': 0.00392156862745098, 'rotation_range': 20, 'width_shift_range': 0.2, 'height_shift_range': 0.2, 'zoom_range': 0.2, 'horizontal_flip': True, 'brightness_range': [0.9, 1.1], 'fill_mode': 'nearest'}

# Class weights
CLASS_WEIGHTS = {0: np.float64(0.5163404114134041), 1: np.float64(18.863636363636363), 2: np.float64(0.9945678223358364), 3: np.float64(0.9998393832316094), 4: np.float64(0.9876249405045217), 5: np.float64(0.9773904851625059), 6: np.float64(0.9910842222576023), 7: np.float64(1.0518756336600203), 8: np.float64(0.9824810606060606), 9: np.float64(1.0345687219544624), 10: np.float64(0.975705329153605)}
USE_CLASS_WEIGHTS = True

# Training parameters
VALIDATION_SPLIT = 0.2
