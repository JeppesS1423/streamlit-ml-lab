import pandas as pd
from pydantic import BaseModel

class CreateDummiesParams(BaseModel):
    drop_first: bool = False
    dummy_na: bool = True
    target_column: str | None = None

def create_dummies(
    data: pd.DataFrame,
    params: CreateDummiesParams,
) -> pd.DataFrame:
    """Create dummy variables from categorical columns.
    
    Args:
        data: Dataframe to work on.
        params: Parameters for dummy variable creation (CreateDummiesParams).
        
    Returns:
        pd.DataFrame: Dataframe with dummy variables created from categorical columns.
    """
    # Exclude target column from dummy creation
    if params.target_column and params.target_column in data.columns:
        # Create dummies for all columns except target
        features = data.drop(columns=[params.target_column])
        target = data[params.target_column]
        
        # Create dummies for features
        features_dummies = pd.get_dummies(
            features,
            drop_first=params.drop_first,
            dummy_na=params.dummy_na
        )
        
        # Convert target to numeric labels if it's categorical
        if target.dtype == 'object' or target.dtype.name == 'category':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            target_encoded = le.fit_transform(target)
            # Store the label encoder for later use if needed
            target_series = pd.Series(target_encoded, name=params.target_column, index=target.index)
        else:
            target_series = target
            
        # Combine features and target
        result = pd.concat([features_dummies, target_series], axis=1)
        return result
    else:
        # Original behavior if no target column specified
        return pd.get_dummies(
            data,
            drop_first=params.drop_first,
            dummy_na=params.dummy_na
        )