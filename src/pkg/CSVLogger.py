import os
import pandas as pd

from typing import Optional, Union, Dict


class CSVLogger:
    def __init__(self, metrics: list, output_dir: str = ".", batch_size: int = 100):
        self.counter = 1
        self.output_dir = os.path.join(output_dir, "log.csv")
        self.csv_file = pd.DataFrame(columns=["epoch"] + metrics)
        self.batch_size = batch_size
        self.rows_to_save = []

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(
        self,
        exc_type: Optional[Exception],
        exc_value: Optional[Exception],
        traceback: Optional[Exception],
    ) -> None:
        self._save()

    def _append(self, row: Dict[str, Union[int, float]] = {}) -> None:
        """Append a row of data to the log."""
        row["epoch"] = self.counter
        if all(key in row for key in self.csv_file.columns):
            self.rows_to_save.append(row)
            self.counter += 1
            if len(self.rows_to_save) >= self.batch_size:
                self._save()
        else:
            print("Row could not be added!\n", row)

    def _save(self) -> None:
        """Save the logged data to a CSV file."""
        if self.rows_to_save:
            df_to_save = pd.DataFrame(self.rows_to_save)
            df_to_save.to_csv(
                self.output_dir,
                mode="a",
                header=not os.path.exists(self.output_dir),
                index=False,
            )
            self.rows_to_save = []

    def log(self, row: Dict[str, Union[int, float]], on_error: str = "print") -> None:
        """Log a row of data.

        Args:
            row (dict): The row of data to log.
            on_error (str, optional): How to handle errors.
                'print' (default) prints to the console, 'raise' raises an exception,
                or 'ignore' to silently ignore errors.
        """
        try:
            self._append(row)
        except Exception as e:
            if on_error == "print":
                print(f"Error logging row: {e}")
            elif on_error == "raise":
                raise
