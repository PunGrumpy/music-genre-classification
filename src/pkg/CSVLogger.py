import pandas as pd
import os


class CSVLogger:
    def __init__(self, metrics, output_dir=".", batch_size=100):
        self.counter = 1
        self.output_dir = os.path.join(output_dir, "log.csv")
        self.csv_file = pd.DataFrame(columns=["epoch"].extend(metrics))
        self.batch_size = batch_size
        self.rows_to_save = []  # Store rows for batching

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._save()  # Save on exit

    def _append(self, row={}):
        row["epoch"] = self.counter
        if all(key in row for key in self.csv_file.columns):
            self.rows_to_save.append(row)
            self.counter += 1
            if len(self.rows_to_save) >= self.batch_size:
                self._save()
        else:
            print("Row could not add!\n", row)  # Or handle differently

    def _save(self):
        if self.rows_to_save:
            df_to_save = pd.DataFrame(self.rows_to_save)
            df_to_save.to_csv(
                self.output_dir,
                mode="a",
                header=not os.path.exists(self.output_dir),
                index=False,
            )
            self.rows_to_save = []

    def log(self, row, on_error="print"):
        """Logs a row of data.

        Args:
            row (dict): The row of data to log.
            on_error (str, optional): How to handle errors.
                'print' (default) prints to console, 'raise' raises an exception,
                or 'ignore' to silently ignore errors.
        """
        try:
            self._append(row)
        except Exception as e:
            if on_error == "print":
                print(f"Error logging row: {e}")
            elif on_error == "raise":
                raise
