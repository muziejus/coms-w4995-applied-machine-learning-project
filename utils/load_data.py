"""
# Load data from the data folder

Moacir P. de Sá Pereira

This function returns a pandas dataframe with the data from the parquet
file for the target company.
"""


def load_data(company):
    return pd.read_parquet(f"data/{company}_merged_data.parquet")
