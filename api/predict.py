import json
import joblib
import numpy as np
import os
from http import HTTPStatus

# Vercel Serverless function handler
def handler(request):
    """Vercel-compatible Python serverless function.

    Expects a POST request with JSON payload: {"features": [f1, f2, f3, f4]}
    Loads the joblib model from the repository and returns the predicted class.
    """
    try:
        if request.method != 'POST':
            return ({'error': 'Only POST supported'}, HTTPStatus.METHOD_NOT_ALLOWED)

        body = request.get_json() if hasattr(request, 'get_json') else json.loads(request.data)
        features = body.get('features')
        if features is None:
            return ({'error': 'Missing "features" in request body'}, HTTPStatus.BAD_REQUEST)

        # Locate model: prefer app/model.joblib then model.joblib at repo root
        possible_paths = [
            os.path.join('app', 'model.joblib'),
            'model.joblib'
        ]
        model_path = None
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            return ({'error': 'Model file not found'}, HTTPStatus.INTERNAL_SERVER_ERROR)

        model = joblib.load(model_path)
        class_names = np.array(['setosa', 'versicolor', 'virginica'])

        arr = np.array(features).reshape(1, -1)
        prediction = model.predict(arr)
        class_name = class_names[prediction][0]

        return ({'predicted_class': str(class_name)}, HTTPStatus.OK)

    except Exception as e:
        return ({'error': str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
