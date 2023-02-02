import sys
import numpy as np
from PIL import Image
from src.constants import *
import tensorflow as tf
from src.logger import logging
from src.exception import CustomException
from src.configuration.s3_syncer import S3Sync
# from tf.keras.preprocessing import image as Img
from src.entity.config_entity import PredictionPipelineConfig, ModelEvaluationConfig


class PredictionPipeline:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.s3 = S3Sync()
        # self.img_size = self.config['data_transformation_config']['img_size']
        self.img_size = self.prediction_pipeline_config.IMG_SIZE

    def image_loader(self, image_bytes):
        """
        Method Name :   image_loader
        Description :   This method load byte image and save it to local.
        Output      :   Returns path the of the saved image
        """
        logging.info("Entered the image_loader method of PredictionPipeline class")
        try:
            logging.info("load byte image and save it to local")
            # input_image = self.config['prediction_pipeline_config']['input_image']
            input_image = self.prediction_pipeline_config.INPUT_IMAGE
            with open(input_image, 'wb') as image:
                image.write(image_bytes)
                image.close()
            path = os.path.join(os.getcwd(), input_image)
            logging.info(f"Returns the saved image: {path}")
            logging.info("Exited the image_loader method of PredictionPipeline class")
            return path
        except Exception as e:
            raise CustomException(e, sys) from e

    # def get_model_from_gcloud(self) -> str:
    #     """
    #     Method Name :   get_model_from_gcloud
    #     Description :   This method fetched the best model from the gcloud.
    #     Output      :   Return best model path
    #     """
    #     logging.info("Entered the get_model_from_gcloud method of PredictionPipeline class")
    #     try:
    #         logging.info("Loading the best model from gcloud bucket")
    #         os.makedirs("artifacts/PredictModel", exist_ok=True)
    #         predict_model_path = os.path.join(os.getcwd(), "artifacts", "PredictModel")
    #         self.gcloud.sync_file_from_gcloud(self.config['prediction_pipeline_config']["bucket_name"],
    #                                           self.config['prediction_pipeline_config']["model_name"],
    #                                           predict_model_path)
    #         best_model_path = os.path.join(predict_model_path, self.config['prediction_pipeline_config']["model_name"])
    #         logging.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
    #         return best_model_path

    #     except Exception as e:
    #         raise CustomException(e, sys) from e

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
            best_model = self.s3.s3_key_path_available(
                                                        bucket_name = self.model_evaluation_config.S3_BUCKET_NAME, 
                                                        s3_key = "ModelTrainerArtifacts/trained_model/"
                                                    )

            if best_model:
                self.s3.sync_folder_from_s3(
                                            folder = self.model_evaluation_config.TRAINED_MODEL_PATH,
                                            bucket_name = self.model_evaluation_config.S3_BUCKET_NAME,
                                            bucket_folder_name = self.model_evaluation_config.BUCKET_FOLDER_NAME
                                            )
            logging.info("Exited the get_model_from_s3 method of PredictionPipeline class")
            best_model_path = os.path.join(self.model_evaluation_config.TRAINED_MODEL_PATH)
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e


    def prediction(self, best_model_path: str, image) -> float:
        """
        Method Name :   prediction
        Description :   This method takes best model path and image
        Output      :   Return the image in base64
        """
        logging.info("Entered the prediction method of PredictionPipeline class")
        try:
            logging.info("Loading best model")
            # model = torch.load(best_model_path, map_location=DEVICE)
            # model.eval()
            model = tf.keras.models.load_model(best_model_path)            

            logging.info("Load the image and preprocess it")

            logging.info("Make the prediction")
            test_image = tf.keras.preprocessing.image.load_img(image, target_size = (224, 224))
            test_image = tf.keras.preprocessing.image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            prediction = model.predict(test_image)
            pred_label = np.argmax(prediction[0])
            # logging.info("Map the predicted label to the corresponding class name")
            predicted_class_name = labels[pred_label.item()]
            logging.info(f'Predicted class name: {predicted_class_name}')
            logging.info("Exited the prediction method of PredictionPipeline class")
            return predicted_class_name

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, data):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            image = self.image_loader(data)
            best_model_path: str = self.get_model_from_s3()
            detected_image = self.prediction(best_model_path, image)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return detected_image
        except Exception as e:
            raise CustomException(e, sys) from e
