import pandas as pd
from pydantic import BaseModel
from typing import Literal, Any

class ScaleFeaturesParams(BaseModel):
    method: Literal["standard", "minmax", "robust"] = "standard"
    train_data: bool = True
    fitted_scaler: Any | None = None

def scale_features(
    data: pd.DataFrame,
    params: ScaleFeaturesParams
) -> tuple[pd.DataFrame, Any | None]:
    """Scale numerical features.

    Args:
        data: Dataframe to work on.
        params: Parameters for feature scaling (ScaleFeaturesParams).

    Returns:
        tuple[pd.DataFrame, Any | None]: Scaled dataframe and fitted scaler (if training).
    """
    
    if params.train_data:
        # Training: fit scaler
        match params.method:
            case "standard":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            case "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            case "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
        
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        return data_scaled, scaler
    
    else:
        # Testing: use fitted scaler
        if params.fitted_scaler is None:
            raise ValueError("fitted_scaler required when train_data=False")
        
        data_scaled = pd.DataFrame(
            params.fitted_scaler.transform(data),
            columns=data.columns,
            index=data.index
        )
        return data_scaled, None