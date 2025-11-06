from http import HTTPStatus

def handler(request):
    """Simple ping endpoint to verify function invocation."""
    return ({'status': 'ok', 'message': 'pong'}, HTTPStatus.OK)
