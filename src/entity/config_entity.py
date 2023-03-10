from dataclasses import dataclass
from src.constants import *
import os 

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME: str = BUCKET_NAME
        self.ZIP_FILE_NAME:str = ZIP_FILE_NAME
        self.S3_DATA_DIR = DATA_DIR
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR,DATA_DIR)
        self.TRAIN_DATA_ARTIFACT_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TRAIN_DIR)
        self.TEST_DATA_ARTIFACT_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TEST_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)
        self.UNZIPPED_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, RAW_FILE_NAME)
        self.ALL_DATASET_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)


@dataclass
class DataTransformationConfig: 
    def __init__(self): 
        self.ROOT_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_TRANSFORM_OBJECT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                                DATA_TRANSFORMATION_TRAIN_FILE_NAME)
        self.TEST_TRANSFORM_OBJECT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                                DATA_TRANSFORMATION_TEST_FILE_NAME)
        self.TRAIN_SPLIT = DATA_TRANSFORMATION_TRAIN_SPLIT
        self.TEST_SPLIT = DATA_TRANSFORMATION_TEST_SPLIT

@dataclass
class ModelTrainerConfig:
     def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR)
        self.BATCH_SIZE: int = TRAINED_BATCH_SIZE
        self.SHUFFLE: bool = TRAINED_SHUFFLE
        self.NUM_WORKERS = TRAINED_NUM_WORKERS
        self.EPOCH: int = EPOCH
        
@dataclass
class ModelEvaluationConfig: 
    def __init__(self):
        self.MODEL_EVALUATION_ARTIFACT_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BUCKET_NAME = BUCKET_NAME 
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.MODEL_EVALUATION_ARTIFACT_DIR,MODEL_DIR)       
        self.S3_MODEL_FOLDER = TRAINED_MODEL_DIR
        self.BUCKET_FOLDER_NAME = BUCKET_FOLDER_NAME
        self.S3_BUCKET_NAME = BUCKET_NAME
        self.MODEL_DIR = MODEL_DIR

    
@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)
        self.BEST_MODEL_PATH: str = os.path.join(self.TRAINED_MODEL_DIR)
        self.BUCKET_NAME: str = BUCKET_NAME
        self.S3_MODEL_KEY_PATH: str = os.path.join(MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)
    
@dataclass
class PredictionPipelineConfig:
    def __init__(self):
        self.INPUT_IMAGE = INPUT_IMAGE
        self.BUCKET_NAME = BUCKET_NAME
        self.THRESHOLD = THRESHOLD
        self.IMG_SIZE = IMG_SIZE