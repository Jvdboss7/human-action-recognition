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