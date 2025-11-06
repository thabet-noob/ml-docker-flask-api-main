import json
import joblib
import numpy as np
import os
import tempfile
import urllib.request
from http import HTTPStatus

# Cached model in global so cold-start load persists across invocations when possible
_MODEL = None
_MODEL_PATH = None

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

        global _MODEL, _MODEL_PATH

        # 1) If MODEL_URL env var provided, download to /tmp once and load
        model_url = os.environ.get('MODEL_URL')
        if model_url:
            if _MODEL is None:
                # download into temp path
                tmpdir = tempfile.gettempdir()
                target = os.path.join(tmpdir, 'model.joblib')
                if not os.path.exists(target):
                    try:
                        urllib.request.urlretrieve(model_url, target)
                    except Exception as e:
                        return ({'error': f'Failed to download model: {str(e)}'}, HTTPStatus.INTERNAL_SERVER_ERROR)
                try:
                    _MODEL = joblib.load(target)
                    _MODEL_PATH = target
                except Exception as e:
                    return ({'error': f'Failed to load model from downloaded file: {str(e)}'}, HTTPStatus.INTERNAL_SERVER_ERROR)
        else:
            # 2) Locate model in repository: prefer app/model.joblib then model.joblib at repo root
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
                return ({'error': 'Model file not found; set MODEL_URL to remote model or include app/model.joblib'}, HTTPStatus.INTERNAL_SERVER_ERROR)

            try:
                _MODEL = joblib.load(model_path)
                _MODEL_PATH = model_path
            except Exception as e:
                return ({'error': f'Failed to load model from repo path: {str(e)}'}, HTTPStatus.INTERNAL_SERVER_ERROR)
        class_names = np.array(['setosa', 'versicolor', 'virginica'])

        arr = np.array(features).reshape(1, -1)
        prediction = _MODEL.predict(arr)
        class_name = class_names[prediction][0]

        return ({'predicted_class': str(class_name)}, HTTPStatus.OK)

    except Exception as e:
        return ({'error': str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
