import re
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Any, Dict, List


def unavailable_to_nan(df: DataFrame, col_list: List[str]) -> None:
    """
    change 'unavailable' to empty string in the specified columns

    Args:
      df: raw DataFrame of attractions
      col_list: list of text columns

    Returns:
      None

    """

    for col in col_list:
        df[col] = df[col].apply(lambda x: np.nan if x == 'unavailable' else x)
        df[col] = df[col].fillna("")


def remove_duplicates_and_nan(df: DataFrame) -> None:
    """
    Remove rows which are exactly the same and

    Args:
      df: DataFrame of attractions

    Returns:
      None

     """
    print("Shape before removing duplicates:", df.shape)
    df.drop_duplicates(subset=['title', 'description', 'address'], inplace=True)
    df.dropna(subset=["text"], inplace=True)
    df.reset_index(inplace=True)
    print("Shape after removing duplicates:", df.shape)


def format_categories(df: pd.DataFrame) -> pd.Series:
    """
    Transforming each tag in "categories_list" column to a list of categories

    Args:
      DataFrame of attractions

    Returns:
      a DataFrame column (Series) with a list of categories in each entry
     """

    return df["categories_list"].apply(
        lambda x: list(set([j.strip().title() for j in re.sub(r'[()\[\'"{}\]]', '', x).strip().split(",")])) if type(
            x) != list else x)


def strip_list(df: DataFrame, col: str):
    """
    Remove empty items from a list of each entry of the prediction column

    Args:
      df: DataFrame with a new column for the different tags_format
      col: str, the name of the new column with the new tags_format

    Returns:
      None
    """
    df[col] = df[col].apply(lambda x: [ele for ele in x if ele.strip()])


def data_preprocess(raw_df: DataFrame) -> DataFrame:
    """
    preprocess the raw DataFrame: update the name of the columns if needed,
    creates 'prediction' column with list of categories,
    creates 'text' column of joining the title and description,
    remove duplicate rows

    Args:
      raw_df: raw DataFrame of attractions

    Returns:
      Pre-processed DataFrame
    """
    raw_df = raw_df.rename(
        columns={"name": "title", "about": "description", "tags": "categories_list", "source": "inventory_supplier",
                 "location_point": "geolocation"})
    if 'prediction' not in raw_df.columns:
        raw_df["prediction"] = format_categories(raw_df)
        strip_list(raw_df, "prediction")
        raw_df["prediction"] = raw_df["prediction"].apply(lambda x: str(x))

    unavailable_to_nan(raw_df, ["title", "description"])
    raw_df["text"] = raw_df["title"] + '. ' + raw_df["description"]
    remove_duplicates_and_nan(raw_df)
    print("The data were processed")
    return raw_df
