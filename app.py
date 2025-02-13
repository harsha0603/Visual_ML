from dash import Dash, dcc, html, Input, Output
from visual import generate_decision_boundary 

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("ML Model Decision Boundary Visualizer"),
    
    dcc.Dropdown(
        id="model-dropdown",
        options=[{"label": key, "value": key} for key in ["SVM", "Decision Tree", "KNN"]],
        value="SVM"
    ),

    dcc.Slider(id="param-slider", min=1, max=10, step=1, value=2, marks={i: str(i) for i in range(1, 11)}),

    dcc.Graph(id="decision-boundary")
])

@app.callback(
    Output("decision-boundary", "figure"),
    [Input("model-dropdown", "value"),
     Input("param-slider", "value")]
)
def update_graph(model_name, param_value):
    return generate_decision_boundary(model_name, param_value)

if __name__ == '__main__':
    app.run_server(debug=True)
