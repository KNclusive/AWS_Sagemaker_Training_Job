# YOLO Training Pipeline on AWS SageMaker: End-to-End ML Lifecycle
This project demonstrates how to build a complete end-to-end machine learning training pipeline on AWS SageMaker using the Ultralytics YOLOv11m model. The pipeline covers data preparation, Docker containerization, model training, and deployment readiness — all orchestrated with AWS services like S3, ECR, and SageMaker.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Dockerfile Details](#dockerfile-details)
- [Why Use SageMaker Standard Directories?](#why-use-sagemaker-standard-directories)
- [Prerequisites](#prerequisites)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [Notes and Best Practices](#notes-and-best-practices)

## Project Overview
This repository illustrates a practical example of training a YOLOv11m object detection model on AWS SageMaker using pre-annotated data stored in S3. The project includes:
- Uploading dataset to S3 via AWS CLI
- Writing and configuring training scripts with Ultralytics YOLO
- Preparing configuration files (`data.yaml` and `training_configuration.yaml`) for SageMaker compatibility
- Building a Docker container following SageMaker conventions
- Pushing the container image to AWS Elastic Container Registry (ECR)
- Creating and running a SageMaker training job with proper IAM roles and resource configurations

## Directory Structure
```
.
├── Dockerfile
├── requirements.txt
├── code/
│   ├── train.py
│   ├── data.yaml
│   └── training_configurations.yaml
├── input-data-config.json
├── output-data-config.json
├── resource-config.json
├── environment.json
├── stopping-condition.json
├── data.yaml
└── training_configuration.yaml
```

## Dockerfile Details
- Base image: `python:3.11`
- Install system dependencies including `libgl1`, `libglib2.0-0` for OpenCV support
- Set working directory to `/opt/ml/code` (SageMaker standard)
- Copy training code and install Python dependencies
- Set entrypoint to `train.py`

## Why Use SageMaker Standard Directories?
SageMaker expects custom containers to follow a specific directory structure for seamless operation and artifact management:
- `/opt/ml/code`: This is the default directory where SageMaker looks for your training script when running a custom container. Setting `WORKDIR /opt/ml/code` ensures SageMaker can find and execute your code without extra configuration.
- Future me: Always put your main training script and related code here for compatibility and easier debugging.
- `/opt/ml/input/data/<channel_name>` (e.g., `/opt/ml/input/data/train`): SageMaker automatically downloads all S3 files specified in your input channels to this directory.
- Future me: Reference your training data and config files using these local paths in your scripts and configs, not S3 URIs. This avoids file-not-found errors and ensures everything is available locally at runtime.
- `/opt/ml/model`: SageMaker expects your training script to write final model artifacts to this directory. After training, SageMaker uploads everything from `/opt/ml/model` to your specified S3 output location.
- Future me: Always save checkpoints, weights, or any output you want persisted to `/opt/ml/model`.

 ## Prerequisites
- AWS Account with permissions for SageMaker, ECR, S3, and IAM
- AWS CLI installed and configured
- Docker installed and configured
- Python 3.11 environment for local development
- Familiarity with SageMaker, Docker, and YOLO training concepts

 ## Step-by-Step Workflow

 ### Upload Data to S3
 Upload your annotated dataset and configuration files to an S3 bucket using AWS CLI:
```
aws s3 cp ./local_data_path s3://<your-bucket>/<your-folder-inside-bucket>/ --recursive
```

### Prepare Training Scripts
- Use Ultralytics YOLO Python API to write your training script (`train.py`).
- Configure `data.yaml` to specify dataset paths.
- Configure `training_configuration.yaml` to specify training parameters and paths, ensuring:
- `project` path is set to `/opt/ml/model` (SageMaker’s model output directory).

### Dockerize the Training Environment
- Create a `Dockerfile` with `WORKDIR /opt/ml/code` as per SageMaker standards.
- Install dependencies from `requirements.txt`.
- Copy your training scripts into the container.
- Ensure your training script (`train.py`) is executable and set as the Docker `ENTRYPOINT`.

### Create ECR Repository
Create an ECR repository to host your Docker image:
```
example:
aws ecr create-repository --repository-name yolo-training-images --region us-east-1
```
To create an Public repository on AWS (Which is supported with upto 50GB cost-free on AWS free-tier
```
aws ecr-public create-repository --repository-name yolo-training-images --region us-east-1
```

### Build and Push Docker Image
Build the Docker image for Linux AMD64 platform (important for SageMaker compatibility):
```
example:
docker buildx build --platform linux/amd64 -t yolo-training:latest .
```
Tag and push the image to ECR:
```
docker tag yolo-training:latest <your-aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/yolo-training-images:latest
```
To know your account id; You can either:
- copy the repositor uri from AWS ECR.
- use command  ```aws sts get-caller-identity --query Account --output text``` given you have configured credentials in CLI.

Authenticate you Docker Client with ECR repository and then push your image
```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.us-east-1.amazonaws.com

docker push <your-aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/yolo-training-images:latest
```

### Configure IAM Role
Create a SageMaker execution role with the following policies:
- `AmazonS3FullAccess`
- `AmazonSageMakerFullAccess`
  
Note the role ARN for use in training job creation.

### Prepare SageMaker Training Job JSON Config Files
Create JSON files for:
- `input-data-config.json`
- `output-data-config.json`
- `resource-config.json`
- `environment.json`
- `stopping-condition.json`
  
These files define the input data channels, output locations, compute resources, environment variables, and job stopping criteria.

### Launch SageMaker Training Job
Use AWS CLI to create the training job:
```
aws sagemaker create-training-job \
  --training-job-name yolo-training-$(date +%Y-%m-%d-%H-%M-%S) \
  --algorithm-specification TrainingImage=<your-aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/yolo-training-images:latest, TrainingInputMode=File \
  --role-arn arn:aws:iam::<your-aws-account-id>:role/sagemaker_role \
  --input-data-config file://input-data-config.json \
  --output-data-config file://output-data-config.json \
  --resource-config file://resource-config.json \
  --environment file://environment.json \
  --stopping-condition file://stopping-condition.json \
  --region us-east-1
```

## Notes and Best Practices
- Always build Docker images with `--platform linux/amd64` on non-Linux hosts to ensure compatibility with SageMaker.
- Store all training input data and config files in S3 and specify them as SageMaker input channels.
- Use `/opt/ml/model` inside the container to save model artifacts for SageMaker to upload post-training.
- Reference all data and config files using local paths inside `/opt/ml/input/data/<channel_name>` in your scripts and configs, not S3 URIs.
- Attach least-privilege IAM policies to SageMaker execution roles.
- Monitor training job logs in CloudWatch for troubleshooting.
