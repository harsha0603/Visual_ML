from dash import Dash, html, dcc
from visual import visualize_linear_regression  # Import visualization function

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Linear Regression Visualization"),
    dcc.Graph(figure=visualize_linear_regression())  # Call the function here
])

if __name__ == "__main__":
    app.run_server(debug=True)
