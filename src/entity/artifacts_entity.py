from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    train_file_path: str
    test_file_path: str
    all_dataset_file_path:str
    
# Data Transformation artifacts
@dataclass
class DataTransformationArtifacts:
    transformed_train_object: str 
    transformed_test_object: str
    train_data_path: str
    test_data_path: str

# Model Trainer artifacts
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    test_dataset: str

# Model evaluation artifacts
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool
    all_losses: str

# Model Pusher Artifacts
@dataclass
class ModelPusherArtifacts:
    bucket_name: str
    s3_model_path: str