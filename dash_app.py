"""
Dash Web Application for ML Model Inference
Interactive web interface to input iris features and get predictions
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import requests
import json
import plotly.graph_objs as go

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Iris Classification Predictor", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    html.Div([
        html.Div([
            html.H3("Enter Iris Features", style={'color': '#34495e'}),
            
            html.Div([
                html.Label("Sepal Length (cm):", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='sepal-length',
                    type='number',
                    placeholder='e.g., 5.1',
                    value=5.1,
                    step=0.1,
                    min=0,
                    max=10,
                    style={'width': '100%', 'padding': '8px', 'margin': '5px 0'}
                )
            ], style={'margin': '10px 0'}),
            
            html.Div([
                html.Label("Sepal Width (cm):", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='sepal-width',
                    type='number',
                    placeholder='e.g., 3.5',
                    value=3.5,
                    step=0.1,
                    min=0,
                    max=10,
                    style={'width': '100%', 'padding': '8px', 'margin': '5px 0'}
                )
            ], style={'margin': '10px 0'}),
            
            html.Div([
                html.Label("Petal Length (cm):", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='petal-length',
                    type='number',
                    placeholder='e.g., 1.4',
                    value=1.4,
                    step=0.1,
                    min=0,
                    max=10,
                    style={'width': '100%', 'padding': '8px', 'margin': '5px 0'}
                )
            ], style={'margin': '10px 0'}),
            
            html.Div([
                html.Label("Petal Width (cm):", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='petal-width',
                    type='number',
                    placeholder='e.g., 0.2',
                    value=0.2,
                    step=0.1,
                    min=0,
                    max=10,
                    style={'width': '100%', 'padding': '8px', 'margin': '5px 0'}
                )
            ], style={'margin': '10px 0'}),
            
            html.Button(
                'Predict Class',
                id='predict-button',
                n_clicks=0,
                style={
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '16px',
                    'marginTop': '20px',
                    'width': '100%'
                }
            ),
            
        ], style={
            'width': '45%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'backgroundColor': '#ecf0f1',
            'borderRadius': '10px',
            'margin': '10px'
        }),
        
        html.Div([
            html.H3("Prediction Result", style={'color': '#34495e'}),
            html.Div(id='prediction-output', style={
                'fontSize': '24px',
                'fontWeight': 'bold',
                'padding': '20px',
                'backgroundColor': '#ffffff',
                'borderRadius': '5px',
                'border': '2px solid #bdc3c7',
                'textAlign': 'center',
                'minHeight': '100px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center'
            }),
            
            html.Div(id='error-output', style={
                'color': '#e74c3c',
                'marginTop': '10px',
                'padding': '10px',
                'backgroundColor': '#fadbd8',
                'borderRadius': '5px',
                'display': 'none'
            }),
            
            html.Div([
                html.H4("Feature Visualization", style={'color': '#34495e', 'marginTop': '30px'}),
                dcc.Graph(id='feature-plot')
            ])
            
        ], style={
            'width': '45%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'backgroundColor': '#ecf0f1',
            'borderRadius': '10px',
            'margin': '10px'
        })
        
    ], style={'textAlign': 'left'}),
    
    html.Div([
        html.H4("Sample Predictions", style={'color': '#34495e'}),
        html.Button(
            'Load Sample Data',
            id='sample-button',
            n_clicks=0,
            style={
                'backgroundColor': '#27ae60',
                'color': 'white',
                'padding': '8px 16px',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'marginBottom': '10px'
            }
        ),
        html.Div(id='sample-output')
    ], style={
        'margin': '20px',
        'padding': '20px',
        'backgroundColor': '#ecf0f1',
        'borderRadius': '10px'
    })
])

# Callback for making predictions
@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-output', 'style'),
     Output('error-output', 'children'),
     Output('error-output', 'style'),
     Output('feature-plot', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [State('sepal-length', 'value'),
     State('sepal-width', 'value'),
     State('petal-length', 'value'),
     State('petal-width', 'value')]
)
def predict_class(n_clicks, sepal_length, sepal_width, petal_length, petal_width):
    default_style = {
        'fontSize': '24px',
        'fontWeight': 'bold',
        'padding': '20px',
        'backgroundColor': '#ffffff',
        'borderRadius': '5px',
        'border': '2px solid #bdc3c7',
        'textAlign': 'center',
        'minHeight': '100px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center'
    }
    
    error_hidden = {'color': '#e74c3c', 'marginTop': '10px', 'padding': '10px', 
                   'backgroundColor': '#fadbd8', 'borderRadius': '5px', 'display': 'none'}
    
    # Create feature visualization
    features = [sepal_length or 0, sepal_width or 0, petal_length or 0, petal_width or 0]
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    fig = go.Figure(data=[
        go.Bar(x=feature_names, y=features, 
               marker_color=['#3498db', '#e74c3c', '#f39c12', '#27ae60'])
    ])
    fig.update_layout(
        title="Input Features",
        yaxis_title="Value (cm)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    if n_clicks == 0:
        return "Click 'Predict Class' to get prediction", default_style, "", error_hidden, fig
    
    # Validate inputs
    if any(val is None for val in [sepal_length, sepal_width, petal_length, petal_width]):
        error_msg = "Please fill in all feature values"
        error_visible = {'color': '#e74c3c', 'marginTop': '10px', 'padding': '10px', 
                        'backgroundColor': '#fadbd8', 'borderRadius': '5px', 'display': 'block'}
        return "Invalid Input", default_style, error_msg, error_visible, fig
    
    try:
        # Prepare data for API call
        url = 'http://0.0.0.0:8000/predict/'
        payload = {'features': [sepal_length, sepal_width, petal_length, petal_width]}
        
        # Make API request
        response = requests.post(url, data=json.dumps(payload), 
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            predicted_class = result['predicted_class']
            
            # Color coding for different classes
            class_colors = {
                'setosa': '#27ae60',
                'versicolor': '#f39c12', 
                'virginica': '#e74c3c'
            }
            
            success_style = default_style.copy()
            success_style['color'] = class_colors.get(predicted_class, '#2c3e50')
            success_style['border'] = f'3px solid {class_colors.get(predicted_class, "#2c3e50")}'
            
            return f"🌸 {predicted_class.title()}", success_style, "", error_hidden, fig
        else:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            error_visible = {'color': '#e74c3c', 'marginTop': '10px', 'padding': '10px', 
                           'backgroundColor': '#fadbd8', 'borderRadius': '5px', 'display': 'block'}
            return "Prediction Failed", default_style, error_msg, error_visible, fig
            
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to ML API. Make sure the Docker container is running on port 8000."
        error_visible = {'color': '#e74c3c', 'marginTop': '10px', 'padding': '10px', 
                        'backgroundColor': '#fadbd8', 'borderRadius': '5px', 'display': 'block'}
        return "Connection Error", default_style, error_msg, error_visible, fig
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        error_visible = {'color': '#e74c3c', 'marginTop': '10px', 'padding': '10px', 
                        'backgroundColor': '#fadbd8', 'borderRadius': '5px', 'display': 'block'}
        return "Error", default_style, error_msg, error_visible, fig

# Callback for loading sample data
@app.callback(
    Output('sample-output', 'children'),
    [Input('sample-button', 'n_clicks')]
)
def load_sample_data(n_clicks):
    if n_clicks == 0:
        return ""
    
    sample_data = [
        [4.3, 3.0, 1.1, 0.1],
        [5.8, 4.0, 1.2, 0.2],
        [5.7, 4.4, 1.5, 0.4],
        [5.4, 3.9, 1.3, 0.4],
        [5.1, 3.5, 1.4, 0.3]
    ]
    
    try:
        url = 'http://0.0.0.0:8000/predict/'
        results = []
        
        for features in sample_data:
            payload = {'features': features}
            response = requests.post(url, data=json.dumps(payload),
                                   headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                prediction = response.json()['predicted_class']
                results.append(html.Div([
                    html.Span(f"Features: {features} → ", style={'fontFamily': 'monospace'}),
                    html.Span(f"{prediction.title()}", 
                             style={'fontWeight': 'bold', 'color': '#2c3e50'})
                ], style={'margin': '5px 0', 'padding': '5px', 'backgroundColor': '#ffffff', 'borderRadius': '3px'}))
            else:
                results.append(html.Div(f"Error for {features}: {response.status_code}"))
        
        return results
        
    except Exception as e:
        return html.Div(f"Error loading samples: {str(e)}", style={'color': '#e74c3c'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
