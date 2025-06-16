import pandas as pd
from pydantic import BaseModel

class CreateDummiesParams(BaseModel):
    drop_first: bool = False
    dummy_na: bool = True

def create_dummies(
    data: pd.DataFrame,
    params: CreateDummiesParams
) -> pd.DataFrame:
    """Create dummy variables from categorical columns.

    Args:
        data: Dataframe to work on.
        params: Parameters for dummy variable creation (CreateDummiesParams).

    Returns:
        pd.DataFrame: Dataframe with dummy variables created from categorical columns.
    """
        
    return pd.get_dummies(
        data,
        drop_first=params.drop_first,
        dummy_na=params.dummy_na
    )