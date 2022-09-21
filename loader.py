from pathlib import Path

import pandas as pd


def load_pokemon(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.drop("#", axis=1, inplace=True)
    return df
