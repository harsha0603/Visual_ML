from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import plotly.graph_objects as go

# Generate classification dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                           n_redundant=0, n_classes=2, random_state=42)

# Dictionary of models
models = {
    "SVM": SVC,
    "Decision Tree": DecisionTreeClassifier,
    "KNN": KNeighborsClassifier
}

def generate_decision_boundary(model_name, param_value):
    """
    Generate decision boundary visualization for the selected model.
    
    Args:
        model_name (str): Model to use ('SVM', 'Decision Tree', or 'KNN').
        param_value (int or float): Parameter for the model.
    
    Returns:
        fig (plotly.graph_objects.Figure): Interactive decision boundary plot.
    """
    
    # Dictionary for dynamic parameter assignment
    params = {}
    if model_name == "SVM":
        params["C"] = param_value
    elif model_name == "Decision Tree":
        params["max_depth"] = param_value
    elif model_name == "KNN":
        params["n_neighbors"] = param_value

    # Initialize and train model
    model = models[model_name](**params)
    model.fit(X, y)

    # Create meshgrid
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    )

    # Predict on grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create figure
    fig = go.Figure()

    # Add contour plot (Decision boundary)
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale='Viridis', opacity=0.7, showscale=False
    ))

    # Add scatter plot (Data points)
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(color=y, colorscale='Jet', size=10)
    ))

    return fig
