import numpy as np
import plotly.graph_objects as go 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples= 100, n_features=1, random_state=42)
X = X.flatten()

model = LinearRegression()
model.fit(X.reshape(-1, 1),y)

X_range = np.linspace(X.min(), X.max(), 100)
y_pred = model.predict(X_range.reshape(-1,1))

