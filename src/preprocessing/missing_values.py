import pandas as pd
from pydantic import BaseModel
from typing import Literal, Any

class HandleMissingValuesParams(BaseModel):
    method: Literal["drop", "mean", "median", "mode", "adaptive", "knn"]
    train_data: bool = True
    fitted_imputer: Any | None = None
    n_neighbors: int | None = 5

def handle_missing_values(
    data: pd.DataFrame, 
    params: HandleMissingValuesParams
) -> tuple[pd.DataFrame, Any | None]:
    """Handle missing values in a dataframe.

    Args:
        data: Dataframe to work on.
        params: Parameters for missing value handling (HandleMissingValuesParams).

    Returns:
        tuple[pd.DataFrame, Any | None]: Processed dataframe and fitted imputer (if applicable).
    """
    
    data_copy = data.copy()
    
    # Drop rows and columns with >=75% null values
    row_thresh = len(data_copy.columns) // 4 + 1
    data_copy = data_copy.dropna(axis=0, thresh=row_thresh)
    col_thresh = len(data_copy) // 4 + 1
    data_copy = data_copy.dropna(axis=1, thresh=col_thresh)
    
    # Handle remaining null values based on chosen method
    match params.method:
        case "drop":
            return data_copy.dropna(axis=0, how="any"), None
            
        case "mean":
            return data_copy.fillna(data_copy.mean()), None
            
        case "median":
            return data_copy.fillna(data_copy.median()), None
            
        case "mode":
            return data_copy.fillna(data_copy.mode().iloc[0]), None
            
        case "adaptive":
            # Use median if highly skewed, mean if approximately normal (for numeric columns only)
            # For categorical columns, use mode
            fill_values = {}
            for col in data_copy.columns:
                if data_copy[col].notna().sum() > 0:  # Only if column has non-null values
                    # Check if column is numeric
                    if data_copy[col].dtype in ["int64", "float64", "int32", "float32"]:
                        skewness = abs(data_copy[col].skew())
                        if skewness > 1:
                            fill_values[col] = data_copy[col].median()
                        else:
                            fill_values[col] = data_copy[col].mean()
                    else:
                        # For categorical columns, use mode
                        fill_values[col] = data_copy[col].mode().iloc[0] if len(data_copy[col].mode()) > 0 else "Unknown"
                else:
                    fill_values[col] = 0 if data_copy[col].dtype in ["int64", "float64", "int32", "float32"] else "Unknown"
            
            return data_copy.fillna(fill_values), None
            
        case "knn":
            from sklearn.impute import KNNImputer
            if params.train_data:
                # Training: fit and transform
                imputer = KNNImputer(n_neighbors=params.n_neighbors)
                data_imputed = pd.DataFrame(
                    imputer.fit_transform(data_copy),
                    columns=data_copy.columns,
                    index=data_copy.index
                )
                return data_imputed, imputer
            
            else:
                # Testing: transform only
                if params.fitted_imputer is None:
                    raise ValueError("fitted_imputer required when train_data=False")
                
                data_imputed = pd.DataFrame(
                    params.fitted_imputer.transform(data_copy),
                    columns=data_copy.columns,
                    index=data_copy.index
                )
                return data_imputed, None