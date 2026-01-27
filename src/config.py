import os

class Config:
    # Data parameters
    IMAGE_SIZE = 256
    NUM_CLASSES = 1  # Only detecting rectangles
    BATCH_SIZE = 32
    
    # Model parameters
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "data/datasets/rectangles")
    DATA_GENERATOR_PATH = os.path.join(BASE_DIR, "data", "DataGenerator.py")
    SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Training parameters
    VAL_SPLIT = 0.2
    NUM_WORKERS = 4
    PIN_MEMORY = True
