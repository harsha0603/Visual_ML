from flask import Flask, render_template
import dash
from dash import dcc, html


app = Flask(__name__)

dash_app = dash.Dash(__name__, server=app, url_base_pathname="/dash/")

dash_app.layout = html.Div([
    html.H1("Ml model visualizer"),
    dcc.Graph(id="grph-layout")
])

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)