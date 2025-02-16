import pandas as pd
import numpy as np
from scipy.stats import skew

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.imputation_log = []  # To store messages for the user
    
    def handle_missing_values(self):
        """
        Automatically detects the best imputation method and applies it.
        """
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:  # Check if column has missing values
                
                if self.df[column].dtype == 'O':  # Categorical column
                    mode_value = self.df[column].mode()[0]
                    self.df[column].fillna(mode_value, inplace=True)
                    self.imputation_log.append(f"Column '{column}' (categorical): Mode imputation applied ({mode_value}).")
                else:  # Numerical column
                    skewness = skew(self.df[column].dropna())  # Calculate skewness
                    
                    if abs(skewness) < 0.5:  # Normally distributed
                        mean_value = self.df[column].mean()
                        self.df[column].fillna(mean_value, inplace=True)
                        self.imputation_log.append(f"Column '{column}' (numerical): Mean imputation applied ({mean_value:.2f}).")
                    else:  # Skewed data, use median
                        median_value = self.df[column].median()
                        self.df[column].fillna(median_value, inplace=True)
                        self.imputation_log.append(f"Column '{column}' (numerical): Median imputation applied ({median_value:.2f}).")
        
        return self.df, self.imputation_log
    
    def get_missing_data_summary(self):
        """
        Returns a summary of missing values in the dataset.
        """
        missing_summary = self.df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        return missing_summary if not missing_summary.empty else "No missing values detected."
