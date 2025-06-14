import pandas as pd
import logging
from typing import Literal

logger = logging.getLogger(__name__)

class DataManager:
    """Dataset manager class including helper functions."""

    def __init__(self, dataset_filepath: str):
        self.dataset = pd.read_csv(dataset_filepath)

    def preprocess(self) -> dict:
        """Prepare the dataset for model training."""
        self.remove_duplicates()

        return {"message": "Dataset successfully prepared for model training"}

    def remove_duplicates(
        self,
        subset: list[str] | str | None = None,
        keep: Literal["first", "last", False] | str = "first",
    ) -> pd.DataFrame:
        """Remove duplicate rows from the dataset.

        Args:
            subset: Column name(s) to consider for identifying duplicates.
            keep: Determines which duplicates to keep.

        Returns:
            pd.DataFrame: The dataset with duplicates removed.
        """        
        # Validate keep parameter
        if keep not in ["first", "last", False]:
            raise ValueError("keep must be 'first', 'last', or False")
        
        # Validate subset columns exist
        if subset is not None:
            if isinstance(subset, str):
                subset = [subset]
            missing_cols = set(subset) - set(self.dataset.columns)
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
        
        # Remove duplicates
        initial_row_count = len(self.dataset)
        self.dataset = self.dataset.drop_duplicates(subset=subset, keep=keep)
        removed_row_count = initial_row_count - len(self.dataset)
        logging.info(f"Removed {removed_row_count} duplicate rows")

        return self.dataset

    def handle_missing_values(self, method: str):
        """Handle missing values in the dataset"""
        # Drop completely empty rows
        initial_row_count = len(self.dataset)
        self.dataset = self.dataset.dropna(axis=0, how="all")

        # Drop rows with >=75% null values
        thresh = len(self.dataset.columns) // 4 + 1
        self.dataset = self.dataset.dropna(axis=0, thresh=thresh)


