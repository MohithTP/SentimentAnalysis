import boto3
import json
import flask
from flask_cors import CORS

# --- CONFIGURATION ---
REGION = 'ap-south-1'
ENDPOINT_PREFIX = 'sentiment-analysis-v2-'

app = flask.Flask(__name__)
CORS(app)

# AWS Clients
sm_client = boto3.client('sagemaker', region_name=REGION)
runtime = boto3.client('sagemaker-runtime', region_name=REGION)

def get_latest_endpoint():
    try:
        response = sm_client.list_endpoints(
            SortBy='CreationTime',
            SortOrder='Descending',
            NameContains='sentiment-analysis' 
        )
        
        endpoints = response.get('Endpoints', [])
        
        if not endpoints:
            return None
        
        for ep in endpoints:
            if ep['EndpointStatus'] == 'InService':
                return ep['EndpointName']
                
        return None
    except Exception as e:
        print(f"Error fetching latest endpoint: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    latest_ep = get_latest_endpoint()
    if latest_ep:
        return f"Proxy is Live! Currently routing to: <b>{latest_ep}</b>"
    return "Proxy is Live, but no 'InService' SageMaker endpoints were found."

@app.route('/invocations', methods=['POST'])
def proxy():
    try:
        # 1. Dynamically find the latest endpoint
        active_endpoint = get_latest_endpoint()
        
        if not active_endpoint:
            return flask.jsonify({'error': 'No active SageMaker endpoint found.'}), 503
            
        print(f"Routing request to: {active_endpoint}")
        data = flask.request.get_json()
        
        # 2. Forward to the dynamic endpoint
        response = runtime.invoke_endpoint(
            EndpointName=active_endpoint,
            ContentType='application/json',
            Body=json.dumps(data)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        # LOGGING FOR EVIDENTLY AI
        try:
            import os
            import csv
            log_file = 'prediction_logs.csv'
            file_exists = os.path.isfile(log_file)
            
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['text', 'label']) # Header
                
                # Zip comments and their predicted labels
                for t, l in zip(data.get('text', []), result.get('predictions', [])):
                    writer.writerow([t, l])
            print(f"Logged {len(result.get('predictions', []))} predictions to {log_file}")
        except Exception as log_e:
            print(f"Logging Error: {log_e}")

        return flask.jsonify(result)
        
    except Exception as e:
        print(f"Error: {e}")
        return flask.jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Checking for available SageMaker endpoints...")
    initial_ep = get_latest_endpoint()
    if initial_ep:
        print(f"Ready! Initial target: {initial_ep}")
    else:
        print("Warning: No active endpoints found yet. Waiting for deployment...")
        
    app.run(port=8080, debug=True)