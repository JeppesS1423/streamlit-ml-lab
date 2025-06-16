import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Literal

class HandleOutliersParams(BaseModel):
    method: Literal["iqr", "zscore"]
    action: Literal["remove", "cap"] = "cap"
    train_data: bool = True
    fitted_boundaries: dict | None = None

def handle_outliers(
    data: pd.DataFrame,
    params: HandleOutliersParams
) -> tuple[pd.DataFrame, dict | None]:
    """Handle outliers in numerical columns based on chosen statistical method and action.

    Args:
        data: Dataframe to work on.
        params: Parameters for outlier handling (HandleOutliersParams).

    Returns:
        tuple[pd.DataFrame, dict | None]: Processed dataframe and outlier boundaries (if training).
    """
     
    data_copy = data.copy()
    numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
    
    if params.train_data:
        # Training: Calculate boundaries from training data
        outlier_info = {}
        
        match params.method:
            case "iqr":
                for col in numeric_cols:
                    q1 = data_copy[col].quantile(0.25)
                    q3 = data_copy[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_info[col] = (lower_bound, upper_bound)

            case "zscore":
                for col in numeric_cols:
                    mean = data_copy[col].mean()
                    std = data_copy[col].std()
                
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    outlier_info[col] = (lower_bound, upper_bound)
        
        # Apply action and return boundaries for test data
        processed_data = _apply_outlier_action(data_copy, outlier_info, params.action)
        return processed_data, outlier_info
    
    else:
        # Testing: Use pre-calculated boundaries
        if params.fitted_boundaries is None:
            raise ValueError("fitted_boundaries required when train_data=False")
        
        # Only process columns that exist in both training and test
        common_cols = set(numeric_cols) & set(params.fitted_boundaries.keys())
        outlier_info = {col: params.fitted_boundaries[col] for col in common_cols}
        
        processed_data = _apply_outlier_action(data_copy, outlier_info, params.action)
        return processed_data, None


def _apply_outlier_action(data: pd.DataFrame, outlier_info: dict, action: str) -> pd.DataFrame:
    """Helper function to apply outlier action.

    Args:
        data: Dataframe to process.
        outlier_info: Dictionary containing outlier boundaries for each column.
        action: Action to take on outliers ("remove" or "cap").

    Returns:
        pd.DataFrame: Dataframe with outliers handled according to specified action.
    """  

    match action:
        case "remove":
            outlier_rows = pd.Series([False] * len(data), index=data.index)
            
            for col, (lower_bound, upper_bound) in outlier_info.items():
                if col in data.columns:
                    col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                    outlier_rows = outlier_rows | col_outliers
            
            return data[~outlier_rows]

        case "cap":
            for col, (lower_bound, upper_bound) in outlier_info.items():
                if col in data.columns:
                    data[col] = data[col].clip(lower_bound, upper_bound)
            
            return data