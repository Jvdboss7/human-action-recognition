import os 
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion constants
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = 'human-action-recognition'
ZIP_FILE_NAME = 'dataset.zip'
DATA_DIR = "data"
RAW_FILE_NAME = 'dataset'
DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
DATA_INGESTION_TRAIN_DIR = 'train_data'
DATA_INGESTION_TEST_DIR = 'test_data' 

# Data transformation constants 
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
DATA_TRANSFORMATION_TRAIN_DIR = 'train_data'
DATA_TRANSFORMATION_TEST_DIR = 'test_data'
DATA_TRANSFORMATION_TRAIN_FILE_NAME = "train.pkl"
DATA_TRANSFORMATION_TEST_FILE_NAME = "test.pkl"
DATA_TRANSFORMATION_TRAIN_SPLIT = 'train'
DATA_TRANSFORMATION_TEST_SPLIT = 'test'

# Model Training Constants
MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
TRAINED_MODEL_DIR = 'trained_model'
# TRAINED_MODEL_NAME = 'model.pt'
TRAINED_BATCH_SIZE = 1
TRAINED_SHUFFLE = False
TRAINED_NUM_WORKERS = 2
EPOCH = 1
EFFICIENTNET_V2_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2"
IMAGE_SHAPE=(224,224)

# Model evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
MODEL_EVALUATION_FILE_NAME = 'loss.csv'
BEST_MODEL_DIR = "best_model"
BUCKET_FOLDER_NAME="ModelTrainerArtifacts/trained_model/"
MODEL_DIR = "trained_model"

# Prediction pipeline constants 
INPUT_IMAGE = "input.jpg"
THRESHOLD = 0.3
IMG_SIZE = 224

# AWS CONSTANTS
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-south-1"

APP_HOST = "0.0.0.0"
APP_PORT = 8080

labels= {0: 'calling', 1: 'clapping', 2: 'cycling', 3: 'dancing', 4: 'drinking', 5: 'eating', 6: 'fighting', 7: 'hugging', 8: 'laughing', 9: 'listening_to_music', 10: 'running', 11: 'sitting', 12: 'sleeping', 13: 'texting', 14: 'using_laptop'}
