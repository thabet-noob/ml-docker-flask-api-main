ML model deployment example
===========================

Complete code (including a trained model) to deploy and inference a machine learning model (built on the iris dataset) using Docker and FastAPI.

1. With terminal navigate to the root of this repository
--------------------------------------------------------

2. Start Docker Desktop (macOS)
-------------------------------
.. code-block::

    open -a Docker

Wait for Docker Desktop to fully start before proceeding to the next step.

3. Build docker image
---------------------
.. code-block::

    docker build -t image_name .

4. Run container
----------------
.. code-block::

    # Option 1: Run with automatic cleanup (recommended)
    docker run --rm --name container_name -p 8000:8000 image_name
    
    # Option 2: Run without cleanup (container persists after stopping)
    docker run --name container_name -p 8000:8000 image_name

**Note**: If you get a "container name already in use" error, either:

- Use the ``--rm`` flag (Option 1) for automatic cleanup, or
- Remove existing containers: ``docker rm container_name``
- Or use a different name: ``docker run --name my_ml_app -p 8000:8000 image_name``

5. Output will contain
----------------------
INFO:     Uvicorn running on http://0.0.0.0:8000

Use this url in chrome to see the model frontend;
use http://0.0.0.0:8000/docs for testing the model in the web interface.

6. Query model
--------------
    
 #. Via **Dash Web Interface** (recommended):
        .. code-block::

            # In a new terminal (keep Docker container running)
            python dash_app.py
            
        Then open: http://localhost:8050
    
 #. Via web interface (chrome):
        http://0.0.0.0:8000/docs -> test model
    
 #. Via python client:
        client.py
    
 #. Via curl request:
        .. code-block::

            curl -X POST "http://0.0.0.0:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

Dash Web Application Features
-----------------------------

The **dash_app.py** provides an interactive web interface with:

- **Input Form**: Enter iris features (sepal length/width, petal length/width)
- **Real-time Prediction**: Get instant classification results
- **Visual Feedback**: Feature visualization with bar charts  
- **Sample Data**: Load and test with pre-defined examples
- **Error Handling**: Clear error messages for connection issues
- **Responsive Design**: Clean, user-friendly interface

**Requirements**: Make sure the Docker container is running on port 8000 before starting the Dash app.

Troubleshooting
---------------

**Container name conflicts**:
.. code-block::

    # List all containers
    docker ps -a
    
    # Remove specific container
    docker rm container_name
    
    # Remove all stopped containers
    docker container prune

**Stop running container**:
.. code-block::

    # Stop by name
    docker stop container_name
    
    # Stop by container ID
    docker stop <container_id>

This repository supports a YouTube `video <>`_