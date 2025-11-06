import json
import os
import traceback
from http import HTTPStatus

# Cached model in global so cold-start load persists across invocations when possible
_MODEL = None
_MODEL_PATH = None


# Vercel Serverless function handler
def handler(request):
    """Vercel-compatible Python serverless function.

    Expects a POST request with JSON payload: {"features": [f1, f2, f3, f4]}
    Loads the joblib model from the repository and returns the predicted class.
    This version surfaces tracebacks and prints debug information so Vercel logs
    contain actionable diagnostics when a function invocation fails.
    """
    try:
        print("[predict] handler start")
        print(f"[predict] REQUEST_METHOD={getattr(request, 'method', None)}")

        if request.method != 'POST':
            return ({'error': 'Only POST supported'}, HTTPStatus.METHOD_NOT_ALLOWED)

        try:
            body = request.get_json() if hasattr(request, 'get_json') else json.loads(request.data)
        except Exception:
            tb = traceback.format_exc()
            print(f"[predict] Failed to parse JSON body:\n{tb}")
            return ({'error': 'Invalid JSON body', 'traceback': tb}, HTTPStatus.BAD_REQUEST)

        features = body.get('features')
        if features is None:
            return ({'error': 'Missing "features" in request body'}, HTTPStatus.BAD_REQUEST)

        global _MODEL, _MODEL_PATH

        # Before we import heavy libraries, confirm there's a model to load.
        model_url = os.environ.get('MODEL_URL')
        print(f"[predict] MODEL_URL={'SET' if model_url else 'NOT_SET'}")

        # If no MODEL_URL and no model file in repo, return a clear error without
        # importing heavy packages that might cause process-level failures in the
        # Vercel runtime.
        if not model_url:
            possible_paths = [
                os.path.join('app', 'model.joblib'),
                'model.joblib'
            ]
            found = any(os.path.exists(p) for p in possible_paths)
            print(f"[predict] model files found in repo: {found}")
            if not found:
                msg = (
                    'Model not available: set MODEL_URL to a public HTTPS model URL '
                    'or include app/model.joblib in the repository.'
                )
                print(f"[predict] {msg}")
                return ({'error': msg}, HTTPStatus.INTERNAL_SERVER_ERROR)

        # Import heavy deps inside the handler so import-time failures don't crash the function
        try:
            import joblib
            import numpy as np
            import tempfile
            import urllib.request
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[predict] Import error:\n{tb}")
            return ({'error': f'Import error: {str(e)}', 'traceback': tb}, HTTPStatus.INTERNAL_SERVER_ERROR)

        try:
            # 1) If MODEL_URL env var provided, download to /tmp once and load
            if model_url:
                if _MODEL is None:
                    # download into temp path
                    tmpdir = tempfile.gettempdir()
                    target = os.path.join(tmpdir, 'model.joblib')
                    print(f"[predict] target download path: {target}")
                    if not os.path.exists(target):
                        try:
                            print(f"[predict] Downloading model from {model_url}...")
                            urllib.request.urlretrieve(model_url, target)
                            print("[predict] Download complete")
                        except Exception as e:
                            tb = traceback.format_exc()
                            print(f"[predict] Failed to download model:\n{tb}")
                            return ({'error': f'Failed to download model: {str(e)}', 'traceback': tb}, HTTPStatus.INTERNAL_SERVER_ERROR)
                    try:
                        print(f"[predict] Loading model from {target}")
                        _MODEL = joblib.load(target)
                        _MODEL_PATH = target
                        print(f"[predict] Model loaded from {target}")
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"[predict] Failed to load model from downloaded file:\n{tb}")
                        return ({'error': f'Failed to load model from downloaded file: {str(e)}', 'traceback': tb}, HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                # 2) Locate model in repository: prefer app/model.joblib then model.joblib at repo root
                possible_paths = [
                    os.path.join('app', 'model.joblib'),
                    'model.joblib'
                ]
                model_path = None
                print(f"[predict] Checking repository paths for model: {possible_paths}")
                for p in possible_paths:
                    exists = os.path.exists(p)
                    print(f"[predict] exists({p})={exists}")
                    if exists:
                        model_path = p
                        break

                if model_path is None:
                    msg = 'Model file not found; set MODEL_URL to remote model or include app/model.joblib'
                    print(f"[predict] {msg}")
                    return ({'error': msg}, HTTPStatus.INTERNAL_SERVER_ERROR)

                try:
                    print(f"[predict] Loading model from repo path {model_path}")
                    _MODEL = joblib.load(model_path)
                    _MODEL_PATH = model_path
                    print(f"[predict] Model loaded from {model_path}")
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"[predict] Failed to load model from repo path:\n{tb}")
                    return ({'error': f'Failed to load model from repo path: {str(e)}', 'traceback': tb}, HTTPStatus.INTERNAL_SERVER_ERROR)

            class_names = np.array(['setosa', 'versicolor', 'virginica'])

            arr = np.array(features).reshape(1, -1)
            prediction = _MODEL.predict(arr)
            class_name = class_names[prediction][0]

            print(f"[predict] prediction={prediction}, class_name={class_name}")

            return ({'predicted_class': str(class_name)}, HTTPStatus.OK)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[predict] Unexpected handler error:\n{tb}")
            return ({'error': str(e), 'traceback': tb}, HTTPStatus.INTERNAL_SERVER_ERROR)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[predict] Fatal error outside handler try:\n{tb}")
        return ({'error': str(e), 'traceback': tb}, HTTPStatus.INTERNAL_SERVER_ERROR)
