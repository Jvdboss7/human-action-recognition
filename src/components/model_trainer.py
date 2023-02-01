import os
import sys
import math
import pandas as pd
from src.constants import *
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts

class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts,
                    model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    #creating function to use different models 
    def create_model(self,model_url, num_classes=10):
        try:
            feature_extractor_layer = hub.KerasLayer(model_url,
                                                    trainable=False,
                                                    name='feature_extraction_layer',
                                                    input_shape=IMAGE_SHAPE+(3,))
            
            model = tf.keras.Sequential([
                feature_extractor_layer,
                layers.Dense(num_classes, activation='softmax', name='output_layer')     
            ])

            return model
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            train_data_gen = load_object(self.data_transformation_artifacts.transformed_train_object)

            train_data=train_data_gen.flow_from_directory(self.data_transformation_artifacts.train_data_path,
                                            target_size=(384,384),
                                            batch_size=8,
                                            class_mode='categorical')

            test_data_gen = load_object(self.data_transformation_artifacts.transformed_test_object)

            test_data=test_data_gen.flow_from_directory(self.data_transformation_artifacts.test_data_path,
                                          target_size=(384,384),
                                          batch_size=8,
                                          class_mode='categorical')
            
            # Create model function
            efficientnet_model = self.create_model(model_url=EFFICIENTNET_V2_URL,
                                        num_classes=15)
            # Compile EfficientNet model
            efficientnet_model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])

            # Fit EfficientNet model 
            efficientnet_history = efficientnet_model.fit(train_data, 
                                              epochs=EPOCH,
                                              steps_per_epoch=len(train_data),
                                              validation_data=test_data,
                                              validation_steps=len(test_data),
                                             )

            
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)
            efficientnet_model.save(self.model_trainer_config.TRAINED_MODEL_PATH)

            logging.info(f"Saved the trained model")

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                test_dataset = test_data
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifacts}")

            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e



