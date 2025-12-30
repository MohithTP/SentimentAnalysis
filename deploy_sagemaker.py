import boto3
import sagemaker
from sagemaker.model import Model

# Configuration
REGION = 'ap-south-1' # Change as needed
ROLE_ARN = 'arn:aws:iam::409749468833:role/SageMaker_Execution' # REPLACE THIS
ECR_REPO_NAME = 'sentiment-analysis-repo'
IMAGE_TAG = 'latest'
ENDPOINT_NAME = 'sentiment-analysis-endpoint'
SKIP_BUILD = False # Set to True to skip docker build and only push/deploy

import subprocess
import sys

def build_and_push(boto_session, account_id, region):
    """
    Builds the Docker image and pushes it to ECR.
    """
    ecr_client = boto_session.client('ecr')
    
    # 1. Create Repository if not exists
    try:
        ecr_client.create_repository(repositoryName=ECR_REPO_NAME)
        print(f"Created ECR repository: {ECR_REPO_NAME}")
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"Repository {ECR_REPO_NAME} already exists.")

    # 2. Login to ECR
    print("Logging in to ECR...")
    # Get authorization token
    auth = ecr_client.get_authorization_token()
    token = auth['authorizationData'][0]['authorizationToken']
    # The token is base64 encoded user:password. We don't need to manually decode if using 'aws ecr get-login-password' via subprocess, 
    # but since we are in python, let's use the CLI command wrapper for simplicity as it handles the pipe to docker login well on Windows.
     
    login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    subprocess.check_call(login_cmd, shell=True)
    
    # 3. Build Docker Image
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO_NAME}:{IMAGE_TAG}"
    
    if not SKIP_BUILD:
        print(f"Building Docker image: {image_uri}...")
        # Assumes Dockerfile is in current directory
        # Using --provenance=false to ensure compatibility with SageMaker/ECR (avoids OCI manifest issues)
        subprocess.check_call(f"docker build --provenance=false -t {ECR_REPO_NAME} .", shell=True)
        
        # 4. Tag Image
        print("Tagging image...")
        subprocess.check_call(f"docker tag {ECR_REPO_NAME}:latest {image_uri}", shell=True)
    else:
        print("Skipping Docker build and tag (SKIP_BUILD=True)...")
    
    # 5. Push Image
    print("Pushing image to ECR (this may take a while)...")
    subprocess.check_call(f"docker push {image_uri}", shell=True)
    
    return image_uri

def deploy():
    print("Setting up SageMaker session...")
    boto_session = boto3.Session(region_name=REGION)
    sess = sagemaker.Session(boto_session=boto_session)
    account_id = boto_session.client('sts').get_caller_identity()['Account']
    
    # --- ADDED: Build and Push Step ---
    image_uri = build_and_push(boto_session, account_id, REGION)
    # ----------------------------------
    
    print(f"Using Role ARN: {ROLE_ARN}")

    # 2. Create SageMaker Model
    print("Creating SageMaker Model object...")
    model = Model(
        image_uri=image_uri,
        role=ROLE_ARN,
        sagemaker_session=sess,
        env={
            'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600' 
        }
    )

    # 3. Cleanup existing endpoint/config if exists
    sm_client = boto_session.client('sagemaker')
    
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
        print(f"Deleted existing endpoint config: {ENDPOINT_NAME}")
    except sm_client.exceptions.ClientError:
        pass # Does not exist
        
    try:
        sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Deleted existing endpoint: {ENDPOINT_NAME}")
    except sm_client.exceptions.ClientError:
        pass # Does not exist

    # 4. Deploy
    print("Deploying to SageMaker Endpoint (this takes a few minutes)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium', # Cost-effective for testing
        endpoint_name=ENDPOINT_NAME,
        serializer=sagemaker.serializers.JSONSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer(),
        container_startup_health_check_timeout=300 # Wait up to 5 mins for TF to load
    )

    print(f"Deployment Complete! Endpoint Name: {ENDPOINT_NAME}")
    return predictor

if __name__ == "__main__":
    # Ensure you are logged in to AWS CLI
    try:
        deploy()
    except Exception as e:
        print(f"Deployment failed: {e}")
        with open("error.txt", "w") as f:
            f.write(str(e))
