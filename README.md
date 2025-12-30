# 🎥Sentiment Analysis System

A full-stack sentiment analysis solution that uses a **BiLSTM Deep Learning model** deployed on **AWS SageMaker** to analyze YouTube comments in real-time via a **Chrome Extension**.

## 🌟 Key Features
- **Deep Learning Model**: Bidirectional LSTM trained for 7-class emotion classification (joy, sadness, anger, fear, etc.).
- **Cloud Scale**: Model deployed as a scalable AWS SageMaker endpoint.
- **Real-time Integration**: Chrome Extension for scraping and analyzing YouTube comments instantly.
- **Secure Proxy**: Integrated Python proxy server to handle AWS IAM signatures securely.

## 🏗️ Project Architecture
```mermaid
graph LR
    YT[YouTube Toolbar] --> EXT[Chrome Extension]
    EXT --> PRX[Local Proxy :8080]
    PRX --> AWS[AWS SageMaker Endpoint]
    AWS --> Model[BiLSTM Model]
```

## 🚀 Getting Started

### Prerequisites
- Python 3.13+
- Docker (for deployment)
- AWS CLI configured with a valid IAM user

### 1. Installation
Clone the repository and install dependencies using `uv`:
```powershell
uv sync
```

### 2. Deployment (AWS SageMaker)
To build the Docker image and deploy your model to the cloud:
```powershell
python deploy_sagemaker.py
```
*Note: This will automatically build the image, push to ECR, and create the SageMaker endpoint.*

### 3. Local Proxy Setup
Because browsers cannot securely make calls to AWS IAM-protected endpoints, run the proxy server locally:
```powershell
uv run python sagemaker_proxy.py
```

### 4. Chrome Extension
1. Open Chrome and go to `chrome://extensions/`.
2. Enable **Developer Mode**.
3. Click **Load Unpacked**.
4. Select the `ChromeExtension` folder from this repository.
5. Go to a YouTube video and click **Analyze Comments**!

## 📁 Repository Structure
- `BiLSTM.py`: Model architecture and training logic.
- `DataSetup.py`: Data cleaning and preprocessing pipeline.
- `deploy_sagemaker.py`: AWS deployment automation.
- `inference.py`: Flask-based inference service (runs in Docker).
- `sagemaker_proxy.py`: Local bridge between Extension and AWS.
- `ChromeExtension/`: UI and scraping logic for the browser.

## 🛠️ Built With
- **TensorFlow/Keras**: Deep Learning framework.
- **SageMaker**: Cloud ML hosting.
- **Flask**: Inference and Proxy API.
- **uv**: Python package management.

---
**Status**: Stable & Active
