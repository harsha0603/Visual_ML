import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate sample regression data
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
X = X.flatten()  # Convert to 1D array

# Train a simple linear regression model
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)

# Generate predictions for visualization
X_range = np.linspace(X.min(), X.max(), 100)
y_pred_final = model.predict(X_range.reshape(-1, 1))

# Create frames for animation
frames = []
steps = np.linspace(0, 1, 50)  # More steps for smoother transition
for step in steps:
    y_pred_step = (1 - step) * np.mean(y) + step * y_pred_final  # Move line gradually
    frames.append(go.Frame(
        data=[
            go.Scatter(x=X, y=y, mode="markers", marker=dict(size=8, color="black"), name="Data Points"),
            go.Scatter(x=X_range, y=y_pred_step, mode="lines", line=dict(color="blue", width=2), name="Regression Line")
        ],
        name=str(step)
    ))

# Create initial scatter plot
fig = go.Figure(
    data=[
        go.Scatter(x=X, y=y, mode="markers", marker=dict(size=8, color="black"), name="Data Points"),
        go.Scatter(x=X_range, y=np.full_like(X_range, np.mean(y)), mode="lines", line=dict(color="blue", width=2), name="Regression Line")
    ],
    frames=frames
)

# Add animation settings (slower speed)
fig.update_layout(
    title="Linear Regression Line Fitting",
    xaxis_title="X",
    yaxis_title="Y",
    updatemenus=[{
        "buttons": [
            {"args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}],
             "label": "Play", "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
             "label": "Pause", "method": "animate"}
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }]
)

fig.show()
