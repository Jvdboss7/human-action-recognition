# human-action-recognition
#### Language and Libraries

<p>
<a><img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" alt="python"/></a>
<a><img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/></a>
<a><img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy"/></a>
<a><img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="opencv"/></a>
<a><img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)" alt="docker"/></a>
<a><img src="https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)" alt="aws"/></a>
</p>


## Problem statement

In this project, we have created an API which clasifies different types of actions of human beings. 

## Solution Proposed
The solution proposed for the above problem is that we have used Computer vision to solve the above problem to classify different types of human actions.
We have used the Tensorflow framework to solve the above problem, and the model used is EfficientNet.Then we created an API that takes in the images and classifies different types of human actions. Then we dockerized the application and deployed the model on the AWS cloud.

## Dataset Used

Dataset composed of 15 classes.

## How to run?

### Step 1: Clone the repository
```bash
git clone my repository 
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -p env python=3.8 -y
```

```bash
conda activate env
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Export the  environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

```
Before running server application make sure your `s3` bucket is available and empty

### Step 5 - Run the application server
```bash
python app.py
```

### Step 6. Train application
```bash
http://localhost:8080/train
```

### Step 7. Prediction application
```bash
http://localhost:8080/predict
```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image

```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> . 

```

3. Run the Docker image

```
docker run -d -p 8080:8080 <IMAGEID>
```

👨‍💻 Tech Stack Used
1. Python
2. FastAPI
3. Tensorflow
4. Docker
5. Computer vision

🌐 Infrastructure Required.
1. AWS S3
2. Amazon ECR
3. Amazon EC2
4. CircleCI


## `src` is the main package folder which contains 

**Artifact** : Stores all artifacts created from running the application

**Components** : Contains all components of Deep Learning Project
- DataIngestion
- DataTransformation
- ModelTrainer
- ModelEvaluation
- ModelPusher

**Custom Logger and Exceptions** are used in the project for better debugging purposes.

## Conclusion

We have created an API which classifies different types of human actions.

=====================================================================