import os
import pandas as pd 
from flask import Flask, render_template, request, url_for, jsonify, redirect
from werkzeug.utils import secure_filename
from visual import generate_regression_plot

app  = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok= True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.split(".",1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            df = pd.load_csv(filepath)
            columns = df.columns.tolist()

            return render_template("select_features.html", columns=columns, filename=filename)
        

    return render_template("upload.html")


@app.route("/train", method = ["POST"])
def train_model():
    filename  = request.form["filename"]
    x_feature = request.form["x_feature"]
    y_feature = request.form["y_feature"]
    lr_rate = float(request.form.get("lr_rate", 0.01))
    iterations = int(request.form.get("iterations",100))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"],filename)
    df = pd.read_csv(filepath)
    plot_html = generate_regression_plot(df, x_feature, y_feature, lr_rate, iterations)

    return render_template("visualize.html", plot_html=plot_html)


if __name__ == "__main__":
    app.run(debug=True)