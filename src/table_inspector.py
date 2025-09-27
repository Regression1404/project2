import hbsir
import pandas as pd


class TableInspector:
    def __init__(self, table_name, years="1396-1403", form="normalized"):
        self.table_name = table_name
        self.years = years
        self.form = form
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []

    def load_table(self) -> pd.DataFrame:
        self.df = hbsir.load_table(
            table_name=self.table_name,
            years=self.years,
            form=self.form,
            redownload= True
        )
        print(f"Table '{self.table_name}' loaded with {len(self.df)} rows and {len(self.df.columns)} columns.\n")
        return self.df

    def inspect_columns(self):
        if self.df is None:
            raise ValueError("Table is not loaded. Please call load_table() first.")

        for col in self.df.columns:
            print(f"\n--- {col} ---")
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_cols.append(col)
                print(f"Min: {self.df[col].min()}, Max: {self.df[col].max()}")
            else:
                self.categorical_cols.append(col)
                unique_vals = self.df[col].dropna().unique()
                print(f"Unique values ({len(unique_vals)}): {unique_vals}")

    def encode_categorical(self):
        if self.df is None:
            raise ValueError("Table is not loaded. Please call load_table() first.")

        if not self.categorical_cols:
            print("No categorical columns to encode.")
            return

        self.df = pd.get_dummies(self.df, columns=self.categorical_cols, drop_first=True)
        print(f"\nCategorical columns converted to numeric. New shape: {self.df.shape}")

    def encode_column_to_dummy(self, column_name: str):
        if column_name not in self.df.columns:
            raise ValueError(f"Column {column_name} not found in the table.")

        dummies = pd.get_dummies(self.df[column_name], prefix=column_name)
        self.df = pd.concat([self.df.drop(columns=[column_name]), dummies], axis=1)
        print(f"Column '{column_name}' encoded to {len(dummies.columns)} dummy columns.")

    def drop_columns(self, columns: list[str]):
        self.df.drop(columns=columns, inplace=True, errors='ignore')
        print(f"Dropped columns: {columns}")
