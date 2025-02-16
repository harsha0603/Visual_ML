# import os 
# import pandas as pd 
# from flask import Flask, url_for, redirect, render_template, request
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# ALLOWED_EXTENSIONS = {"csv"}

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".",1)[1].lower()


# @app.route("/", methods = ["GET","POST"])

# def upload_file():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return redirect(request.url)
        
#         file = request.files["file"]
#         if file.filename == "":
#             return redirect(request.url)

#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             file.save(filepath)

#         df = pd.read_csv(filepath)
#         columns = df.columns.tolist()

#         return render_template("select_features.html", columns = columns , filename = filename)

#     return render_template("upload.html")



# Data preprocessing code

import pandas as pd 
import numpy as np 
from scipy.stats import skew

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.imputation_log = []

    def handle_missing_values(self):
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                if self.df[column].dtype == 0: # Categorical column 
                    mode_value  = self.df[column].mode()[0]
                    self.df[column].fillna(mode_value, inplace = True)
                    self.imputation_log.append(f"Column '{column}' (categorical): Mode imputation applied ({mode_value}).")
                else:
                    skewness = skew(self.df[column].dropna()) # Calculate skewness
                    if abs(skewness) < 0.5:
                        mean_value = self.df[column].mean()
                        self.df[column].fillna(mean_value, inplace = True)
                        self.imputation_log.append(f"Column '{column}' (Numerical): Mean imputation applied ({mean_value}).")
                    else:
                        median_value = self.df[column].median()
                        self.df[column].fillna(median_value, inplace = True)
                        self.imputation_log.append(f"Column '{column}' (Numerical): Median imputation applied ({median_value}).")   

        return self.df, self.imputation_log
    
    def get_missing_data_summary(self):
        missing_summary = self.df.isnull().sum()
        missing_summary  = missing_summary[missing_summary > 0]
        return missing_summary if not missing_summary.empty else "No missing data avaliable"