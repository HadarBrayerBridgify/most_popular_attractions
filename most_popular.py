import re
import json
import uuid
import random
import pandas as pd
from pandas import DataFrame
from torch import Tensor
import numpy as np
from typing import Any, Dict, List
import data_preprocessing as dp
import attractions_similarity_updater as attrac_sim

SIMILARITY_THRESHOLD: float = 0.65
ATTRACTIONS_PER_SUPPLIER: int = 40
NUM_ATTRACTIONS_TO_DISPLAY: int = 15
RELEVANT_SUPPLIERS: List[str] = ['Getyourguide', 'Viator', 'Musement', 'Tiqets']


def all_most_popular(attractions: DataFrame) -> List[Dict[str, str]]:
    """
    Extract the specified number of most popular attractions from each supplier
    and join all to one DataFrame

    Args:
      attractions: DataFrame of all the attractions in the database

    Return:
      DataFrame with most popular attractions of each paid supplier
    """

    all_most_popular_attractions = pd.DataFrame()

    for supplier in RELEVANT_SUPPLIERS:
        most_pop_supplier: DataFrame = attractions[attractions["inventory_supplier"] == supplier].sort_values(
            by="number_of_reviews", ascending=False)[:ATTRACTIONS_PER_SUPPLIER]
        most_pop_supplier.sort_values(by="number_of_reviews", ascending=True, inplace=True)
        most_pop_supplier["rank"] = [i for i in range(most_pop_supplier.shape[0])]
        all_most_popular_attractions: DataFrame = pd.concat([all_most_popular_attractions, most_pop_supplier])
        all_most_popular_attractions_dict = all_most_popular_attractions.to_dict('records')
    return all_most_popular_attractions_dict


def choose_x_most_popular_idx(data_with_similarity_id: DataFrame) -> List[int]:
    """
    select the specified number of indices of most popular attractions

    Args:
      data_with_similarity_id: DataFrame of all most popular attractions with similarity_uuid column

    Return:
      list of indices of the chosen most popular attractions
    """

    supplier_count_dict: Dict[str, int] = {supplier: 0 for supplier in RELEVANT_SUPPLIERS}
    popular_data: DataFrame = data_with_similarity_id.copy()
    chosen_idx = list()

    for i in range(NUM_ATTRACTIONS_TO_DISPLAY):
        max_rank: int = popular_data["rank"].max()
        most_popular_df: DataFrame = popular_data[popular_data["rank"] == max_rank]
        pop_suppliers: List[str] = most_popular_df["inventory_supplier"].unique()
        min_count = 100
        for supplier in pop_suppliers:
            if supplier_count_dict[supplier] < min_count:
                min_count = supplier_count_dict[supplier]
                chosen_supplier: str = supplier
                supplier_count_dict[chosen_supplier] += 1

        chosen_attraction_idx: int = random.choice(
            most_popular_df[most_popular_df["inventory_supplier"] == chosen_supplier].index)
        chosen_idx.append(chosen_attraction_idx)
        similarity_group: str = popular_data.loc[chosen_attraction_idx]["similarity_uuid"]

        if not pd.isna(similarity_group):
            popular_data: DataFrame = popular_data[popular_data["similarity_uuid"] != similarity_group]
        else:
            popular_data.drop(index=chosen_attraction_idx, inplace=True)

    return chosen_idx


def create_most_pop_dict(df, chosen_idx):
    """
    Creates a list of dictionaries of uuid: rank (1-15, most_popular_global)

    Args:
      df: DataFrame of all popular attractions with similarity_uuid column
      chosen_idx: list of int, The output of choose_x_most_popular_idx function

    Return:
      list of dictionaries of uuid: rank (1-16, most_popular_global)
    """

    chosen_most_pop_df = pd.DataFrame(df["uuid"].iloc[chosen_idx])
    chosen_most_pop_df["uuid"] = chosen_most_pop_df["uuid"].apply(lambda id: str(id))
    chosen_most_pop_df.rename(columns={'uuid': 'attraction_id'}, inplace=True)
    chosen_most_pop_df["rank"] = [i for i in range(1, NUM_ATTRACTIONS_TO_DISPLAY + 1)]
    return chosen_most_pop_df.to_dict('records')


def selected_most_popular(attractions: List[Dict[str, str]]):
    """
    select the specified number of most popular attractions out of all the attractions in the database

    Args:
      attractions: List of dictionaries of the attractions

    Return:
      List of dictionaries of selected most popular attractions {'uuid':int}
    """

    attractions_df: DataFrame = pd.DataFrame.from_dict(attractions)
    attractions_df_preprocess: DataFrame = dp.data_preprocess(attractions_df)

    all_popular_attractions_dict: List[Dict[str, str]] = all_most_popular(attractions_df_preprocess)
    all_most_popular_attractions_df = pd.DataFrame.from_dict(all_popular_attractions_dict)
    # create similarity groups
    similarity_groups: List[Dict[str, str]] = attrac_sim.compute_similarity_groups(all_popular_attractions_dict)
    similarity_groups_df: DataFrame = pd.DataFrame.from_dict(similarity_groups)

    similarity_groups_df.rename(columns={'id': 'uuid'}, inplace=True)
    data_with_similarity: DataFrame = pd.merge(all_most_popular_attractions_df, similarity_groups_df, how='outer')

    chosen_popular_idx: List[int] = choose_x_most_popular_idx(data_with_similarity)
    most_popular_dict = create_most_pop_dict(data_with_similarity, chosen_popular_idx)

    return most_popular_dict


