import json
import flask
import pandas as pd
import tensorflow as tf
import mlflow.keras
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from DataSetup import clean_text

app = flask.Flask(__name__)

# Load Model and Artifacts Global Variables
model = None
tokenizer = None
le = None
max_len = 100
load_error = None

def load_artifacts():
    global model, tokenizer, le, load_error
    
    print("STARTING LOADING ARTIFACTS...", flush=True)
    
    # 1. Load Model
    
    possible_paths = ['model', '/app/model', '/opt/ml/model']
    model_path = None
    
    for p in possible_paths:
        if os.path.exists(p) and os.path.isdir(p) and len(os.listdir(p)) > 0:
            model_path = p
            break
            
    if not model_path:
        print(f"Error: Could not find model in {possible_paths}", flush=True)
        model_path = 'model'

    print(f"Loading model from: {model_path}", flush=True)
    
    # DEBUG: Print versions
    import keras
    import tensorflow
    print(f"Runtime Keras Version: {keras.__version__}", flush=True)
    print(f"Runtime TensorFlow Version: {tensorflow.__version__}", flush=True)

    try:
        # Check if directory exists
        if not os.path.exists(model_path):
             raise Exception(f"Model directory {model_path} does not exist.")
             
        model = mlflow.keras.load_model(model_path)
        print("Model loaded successfully!", flush=True)
    except Exception as e:
        load_error = f"Model Load Error: {str(e)}"
        print(load_error, flush=True)

    # 2. Load Tokenizer & Encoder
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('label_encoder.pickle', 'rb') as handle:
            le = pickle.load(handle)
        print("Pickles loaded successfully!", flush=True)
    except Exception as e:
        if not load_error:
            load_error = f"Pickle Load Error: {str(e)}"
        print(f"Pickle Load Error: {str(e)}", flush=True)

load_artifacts()

@app.route('/ping', methods=['GET'])
def ping():
    return flask.Response(response='\n', status=200, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Inference endpoint."""
    global load_error
    
    if load_error:
         return flask.Response(response=json.dumps({"error": load_error}), status=500, mimetype='application/json')

    if flask.request.content_type == 'application/json':
        data = flask.request.get_json()
        input_text = data.get('text')
    else:
        return flask.Response(response='This predictor only supports application/json data', status=415, mimetype='text/plain')

    if input_text is None:
        return flask.Response(response='Missing "text" field in JSON', status=400, mimetype='text/plain')
    
    try:
        # Preprocess
        if isinstance(input_text, str):
            input_text = [input_text]
        
        print(f"Processing {len(input_text)} texts...", flush=True)
        cleaned_texts = [clean_text(t) for t in input_text]
        
        # Tokenize
        if not tokenizer:
             raise Exception("Tokenizer not loaded")
             
        sequences = tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(sequences, maxlen=max_len)
        print(f"Input shape for prediction: {padded.shape}", flush=True)
        
        # Predict
        print("Starting prediction...", flush=True)
        preds = model.predict(padded)
        print("Prediction successful!", flush=True)
        pred_indices = np.argmax(preds, axis=1)
        
        if le:
            results = le.inverse_transform(pred_indices).tolist()
        else:
            results = pred_indices.tolist()
            
        return flask.Response(response=json.dumps({"predictions": results}), status=200, mimetype='application/json')
        
    except Exception as e:
        import traceback
        err_msg = f"Inference Logic Error: {str(e)}\n{traceback.format_exc()}"
        print(err_msg, flush=True)
        return flask.Response(response=json.dumps({"error": err_msg}), status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
