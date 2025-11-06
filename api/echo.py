import json
from http import HTTPStatus

def handler(request):
    """Simple echo endpoint for debugging serverless invocation.

    POST JSON body will be returned as-is with status 200.
    """
    try:
        if request.method != 'POST':
            return ({'error': 'Only POST supported'}, HTTPStatus.METHOD_NOT_ALLOWED)

        # Some runtimes provide get_json(), others expose raw data
        try:
            body = request.get_json()
        except Exception:
            body = json.loads(request.data) if request.data else {}

        return (body, HTTPStatus.OK)
    except Exception as e:
        return ({'error': str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
