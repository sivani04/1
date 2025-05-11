import pandas as pd
from Config import *

def get_type_s(df: pd.DataFrame):
        return df[Config.CLASS_COL]

def concat_types(types: list[str]) -> str:
    return Config.JOIN_CHAR.join(types)

def format_types(types: pd.Series) -> pd.Series:
    types = types.str.strip().str.lower().fillna(Config.EMPTY_TYPE)
    return pd.Series([concat_types(types[:i + 1]) for i in range(len(types))])

def parse_full_type(full_type: str) -> pd.Series:
    return format_types(pd.Series(full_type.split(Config.JOIN_CHAR)))