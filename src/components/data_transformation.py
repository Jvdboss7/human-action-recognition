import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.utils.main_utils import save_object
from src.entity.config_entity import DataTransformationConfig
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def image_data_gen(self):
        try:
            train_datagen=ImageDataGenerator(
                                            rescale=1/255., 
                                            shear_range=0.2, 
                                            zoom_range=0.2, 
                                            horizontal_flip=True
                                            )

            test_datagen=ImageDataGenerator(rescale=1/255.)

            # train_data=train_datagen.flow_from_directory(self.data_ingestion_artifact.train_file_path,
            #                                             target_size=(384,384),
            #                                             batch_size=32,
            #                                             class_mode='categorical')
            # test_data=test_datagen.flow_from_directory(self.data_ingestion_artifact.test_file_path,
            #                                             target_size=(384,384),
            #                                             batch_size=32,
            #                                             class_mode='categorical')
            return train_datagen,test_datagen
            
        except Exception as e:
            raise CustomException(e,sys) from e


    def initiate_data_transformation(self) -> DataTransformationArtifacts:

        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")

            train_data,test_data=self.image_data_gen()
            
            save_object(self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH, train_data)
            save_object(self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH, test_data)
            # os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,exist_ok=True)

            logging.info("Saved the train transformed object")

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_train_object=self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH,
                transformed_test_object=self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH
                )

            logging.info(f'{data_transformation_artifact}')

            logging.info("Exited the initiate_data_transformation method of Data transformation class")

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
            