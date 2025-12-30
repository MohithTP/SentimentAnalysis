import boto3
import sagemaker
from sagemaker.model import Model
import os
import subprocess
import time

# --- CONFIGURATION ---
REGION = os.getenv('AWS_REGION', 'ap-south-1')
ROLE_ARN = 'arn:aws:iam::409749468833:role/SageMaker_Execution'
ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID', '409749468833')
ECR_REPO_NAME = 'sentiment-analysis-repo'
IMAGE_TAG = 'latest'

# DYNAMIC NAMING
VERSION = "v2" 
TIMESTAMP = int(time.time())
ENDPOINT_NAME = f"sentiment-analysis-{VERSION}-{TIMESTAMP}"

SKIP_BUILD = False 

def build_and_push(boto_session, account_id, region):
    """ Builds the Docker image and pushes it to ECR. """
    ecr_client = boto_session.client('ecr')
    
    # 1. Create Repository if not exists
    try:
        ecr_client.create_repository(repositoryName=ECR_REPO_NAME)
        print(f"Created ECR repository: {ECR_REPO_NAME}")
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"Repository {ECR_REPO_NAME} already exists.")

    # 2. Login to ECR
    print("Logging in to ECR...")
    login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    subprocess.check_call(login_cmd, shell=True)
    
    # 3. Build Docker Image
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO_NAME}:{IMAGE_TAG}"
    
    if not SKIP_BUILD:
        print(f"Building Docker image: {image_uri}...")
        # Force linux/amd64 and remove provenance to satisfy SageMaker manifest requirements
        build_cmd = (
            f"docker buildx build --platform linux/amd64 --provenance=false "
            f"-t {image_uri} . --load"
        )
        subprocess.check_call(build_cmd, shell=True)
    
    # 5. Push Image
    print("Pushing image to ECR...")
    subprocess.check_call(f"docker push {image_uri}", shell=True)
    
    return image_uri

def deploy():
    print(f"--- Starting Deployment for Endpoint: {ENDPOINT_NAME} ---")
    boto_session = boto3.Session(region_name=REGION)
    sess = sagemaker.Session(boto_session=boto_session)
    account_id = boto_session.client('sts').get_caller_identity()['Account']
    
    # 1. Build and Push
    image_uri = build_and_push(boto_session, account_id, REGION)
    
    # 2. Create SageMaker Model
    model_name = f"sentiment-model-{VERSION}-{TIMESTAMP}"
    print(f"Creating SageMaker Model: {model_name}")
    
    model = Model(
        image_uri=image_uri,
        role=ROLE_ARN,
        sagemaker_session=sess,
        name=model_name, 
        env={
            'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600',
            'PYTHONUNBUFFERED': '1'
        }
    )

    # 3. Deploy
    print(f"Deploying to SageMaker Endpoint: {ENDPOINT_NAME}...")
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large', 
        endpoint_name=ENDPOINT_NAME,
        serializer=sagemaker.serializers.JSONSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer(),
        container_startup_health_check_timeout=600 # 10 minutes for heavy TF models
    )

    print(f"\nSUCCESS!")
    print(f"New Endpoint Name: {ENDPOINT_NAME}")
    print(f"You now have a separate deployment running alongside your old ones.")
    return predictor

if __name__ == "__main__":
    try:
        deploy()
    except Exception as e:
        print(f"Deployment failed: {e}")