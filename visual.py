import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_missing_values_heatmap(df):
    """
    Plots a heatmap showing missing values in the dataset.
    """
    plt.figure(figsize=(10, 6))
    plt.style.use("dark_background")  # Dark theme
    
    sns.heatmap(df.isnull(), cmap="coolwarm", cbar=False, yticklabels=False)
    plt.title("Missing Values Heatmap", fontsize=14, color='white')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Sample dataset with missing values
    data = {
        "Feature1": [1, 2, np.nan, 4, 5, np.nan],
        "Feature2": [np.nan, 2, 3, 4, np.nan, 6],
        "Feature3": [1, 2, 3, 4, 5, 6]
    }
    df = pd.DataFrame(data)
    plot_missing_values_heatmap(df)
