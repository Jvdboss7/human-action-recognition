import os
import sys
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifacts_entity import ModelTrainerArtifacts, DataTransformationArtifacts, ModelEvaluationArtifacts
from src.configuration.s3_syncer import S3Sync


class ModelEvaluation:

    def __init__(self, model_evaluation_config:ModelEvaluationConfig,
                data_transformation_artifacts:DataTransformationArtifacts,
                model_trainer_artifacts:ModelTrainerArtifacts):

        self.model_evaluation_config = model_evaluation_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts
        self.s3 = S3Sync()
        self.bucket_name = BUCKET_NAME

    def get_model_from_s3(self) -> str:
        """
        Method Name :   predict
        Description :   This method predicts the image.

        Output      :   Predictions
        """
        logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")
        try:
            logging.info(f"Checking the s3_key path{self.model_evaluation_config.TRAINED_MODEL_PATH}")
            print(f"s3_key_path:{self.model_evaluation_config.TRAINED_MODEL_PATH}")
            best_model = self.s3.s3_key_path_available(bucket_name=self.model_evaluation_config.S3_BUCKET_NAME,s3_key="ModelTrainerArtifacts/trained_model/")

            if best_model:
                self.s3.sync_folder_from_s3(folder=self.model_evaluation_config.TRAINED_MODEL_PATH,bucket_name=self.model_evaluation_config.S3_BUCKET_NAME,bucket_folder_name=self.model_evaluation_config.BUCKET_FOLDER_NAME)
            logging.info("Exited the get_model_from_s3 method of PredictionPipeline class")
            best_model_path = os.path.join(self.model_evaluation_config.TRAINED_MODEL_PATH)
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
                Method Name :   initiate_model_evaluation
                Description :   This function is used to initiate all steps of the model evaluation

                Output      :   Returns model evaluation artifact
                On Failure  :   Write an exception log and then raise an exception
        """

        try:
            trained_model = tf.keras.models.load_model(self.model_trainer_artifacts.trained_model_path)            
            test_data = self.model_trainer_artifacts.test_dataset
            print(test_data)
            loss = trained_model.evaluate(test_data)
            print(loss)

            os.makedirs(self.model_evaluation_config.TRAINED_MODEL_PATH, exist_ok=True)
            

            s3_model_path = self.get_model_from_s3()

            logging.info(f"{s3_model_path}")

            is_model_accepted = False
            s3model_loss = None 
            print(f"{os.path.isdir(s3_model_path)}")
            if os.path.isdir(s3_model_path) is False: 
                is_model_accepted = True
                print("s3 model is false and model accepted is true")
                s3model_loss = None

            else:
                print("Entered inside the else condition")
                s3_model = tf.keras.models.load_model(s3_model_path)
                print("Model loaded from s3")
                s3model_loss = s3_model.evaluate(test_data)

                if s3model_loss > loss:
                    print(f"printing the loss inside the if condition{s3model_loss} and {loss}")
                    # 0.03 > 0.02
                    is_model_accepted = True
                    print("f{is_model_accepted}")
            model_evaluation_artifact = ModelEvaluationArtifacts(
                        is_model_accepted=is_model_accepted,
                        all_losses=loss)
            print(f"{model_evaluation_artifact}")

            logging.info("Exited the initiate_model_evaluation method of Model Evaluation class")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

