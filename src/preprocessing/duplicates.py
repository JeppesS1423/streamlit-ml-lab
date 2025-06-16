import pandas as pd
from pydantic import BaseModel
from typing import Literal

class RemoveDuplicateParams(BaseModel):
    subset: list[str] | str | None = None
    keep: Literal["first", "last", False] = "first"

def remove_duplicates(
    data: pd.DataFrame,
    params: RemoveDuplicateParams
) -> pd.DataFrame:
    """Remove duplicate rows from a dataframe.

    Args:
        data: Dataframe to work on.
        params: Parameters for duplicate removal (RemoveDuplicateParams)

    Returns:
        pd.DataFrame: Dataframe with duplicates removed.
    """
    
    # Validate subset columns exist
    if params.subset is not None:
        if isinstance(params.subset, str):
            subset = [params.subset]
        missing_cols = set(params.subset) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
    
    result_data = data.drop_duplicates(subset=params.subset, keep=params.keep)

    return result_data