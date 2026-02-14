import os
import json
import joblib
import numpy as np


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, 'model.joblib'))


def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        data = json.loads(request_body)
        if isinstance(data, dict) and 'instances' in data:
            return np.array(data['instances'])
        if isinstance(data, list):
            return np.array(data)
        raise ValueError("Expected 'instances' key or array in JSON body")
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    return {'predictions': predictions.tolist(), 'probabilities': probabilities.tolist()}


def output_fn(prediction, accept='application/json'):
    if accept == 'application/json':
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
