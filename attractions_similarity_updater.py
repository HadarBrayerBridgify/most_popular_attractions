from datetime import datetime, timedelta
from typing import Any, Dict, List, Set
from aws_services.simple_queue_service import SQSQueue
from database import DataBase
import uuid
import torch
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import data_preprocessing as dp
from torch import Tensor
from sentence_transformers import SentenceTransformer, util

QUEUE_URL: str = "url"
CITY_ID: str = "id"
SIMILARITY_THRESHOLD: float = 0.65


def _model_embedding(df: DataFrame, col: str) -> Tensor:
    """
    calculates the embeddings (as torch) of each entry in 'text' column according to SentenceTransformers

    Args:
      df: preprocessed DataFrame
      col: str, the name of the text column according to which the embeddings will be calculated

    Returns:
      torch.Tensor
    """
    model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = df[col].values
    embeddings: Tensor = model.encode(sentences, convert_to_tensor=True)
    print("finished embeddings")
    return embeddings


def _pairs_df_model(embeddings: Tensor) -> DataFrame:
    """
    Compute cosine-similarities of each embedded vector with each other embedded vector

    Args:
      embeddings: Tensor embeddings of the text column

    Returns:
      DataFrame with columns: 'ind1' (vector index), 'ind2' (vector index), 'score' (cosine score of the vectors)
      (The shape of the DataFrame is: rows: (n!/(n-k)!k!), for k items out of n)

    """
    cosine_scores: Tensor = util.cos_sim(embeddings, embeddings)
    pairs: List[Dict[str, Any]] = []
    for i in range(len(cosine_scores) - 1):
        for j in range(i + 1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    pairs_df = pd.DataFrame(pairs)

    pairs_df["ind1"] = pairs_df["index"].apply(lambda x: x[0]).values
    pairs_df["ind2"] = pairs_df["index"].apply(lambda x: x[1]).values

    return pairs_df


def similarity_matrix(similarity_idx_df: DataFrame, reduced_df: DataFrame) -> DataFrame:
    """
    creates n^2 similarity matrix. Each attraction has a similarity score in relation to each attraction in the data

    Args:
      similarity_idx_df: DataFrame output of the function pairs_df_model
      reduced_df: preprocessed DataFrame

    Returns:
      sqaure DataFrame. columns = index = the indices of the attractions. values: similarity score
    """
    similarity_matrix: DataFrame = pd.DataFrame(columns=[i for i in range(reduced_df.shape[0])], index=range(reduced_df.shape[0]))
    for i in range(reduced_df.shape[0]):
        for j in range(i, reduced_df.shape[0]):
            if j == i:
                similarity_matrix.iloc[i][j] = 1
                similarity_matrix.iloc[j][i] = 1
            else:
                similarity_score = \
                    similarity_idx_df[(similarity_idx_df["ind1"] == i) & (similarity_idx_df["ind2"] == j)][
                        "score"].values
                similarity_matrix.iloc[i][j] = similarity_score
                similarity_matrix.iloc[j][i] = similarity_score
    return similarity_matrix


def _groups_idx(similarity_df: DataFrame) -> List[set[int]]:
    """
    Creates a list of sets, each tuple is a similarity group which contains the attractions indices (A group consists
    of the pairs of a particular index and the pairs of its pairs. There is no overlap of indices between the groups

    Args:
      similarity_df: DataFrame output of the function pairs_df_model

    Returns:
      a list of sets. Each tuple contains attractions indices and represent a similarity group
    """
    sets_list: List[set[int]] = list()

    for idx in similarity_df["index"].values:
        was_selected = False
        first_match: set[int] = set()

        for group in sets_list:
            intersec = set(idx) & group
            if len(intersec) > 0:
                group.update(idx)
                first_match.update(group)
                sets_list.remove(group)
                was_selected = True

        if len(first_match) > 0:
            sets_list.append(first_match)

        if not was_selected:
            sets_list.append(set(idx))

    return sets_list


def _groups_df(similarity_df_above_threshold: DataFrame, df: DataFrame) -> List[Dict[str, str]]:
    """
    Creates a DataFrame of 'uuid' and 'similarity_uuid' of the attractions which have similarity score above the threshold

    Args:
      similarity_df_above_threshold: a filtered DataFrame of the output of pairs_df which pass 'score' > threshold
      df: pre-processed DataFrame of the attractions

    Returns:
      a DataFrame of 'uuid' and 'similarity_uuid'
    """
    display_columns: List[str] = ['uuid']
    above_threshold_idx: List[int] = list(set(np.array([idx for idx in similarity_df_above_threshold["index"]]).ravel()))
    df_above_threshold: DataFrame = df.loc[above_threshold_idx][display_columns]
    df_above_threshold.columns = ["id"]
    df_above_threshold['similarity_uuid'] = 0
    groups_list: List[set[int]] = _groups_idx(similarity_df_above_threshold)

    for group in groups_list:
        df_above_threshold['similarity_uuid'].loc[list(group)] = str(uuid.uuid4())

    similarity_groups_json: List[Dict[str, str]] = df_above_threshold.to_dict('records')
    return similarity_groups_json


def compute_similarity_groups(
        attractions: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Creates a similarity uuid for each attraction with similarities

    Args:
        attractions: List of dictionaries of the attractions

    Returns:
        List of dictionaries, each dictionary contains "uuid" : "similarity_uuid"
    """
    raw_df: DataFrame = pd.DataFrame.from_dict(attractions)
    df_reduced: DataFrame = dp.data_preprocess(raw_df)

    embeddings_text: Tensor = _model_embedding(df_reduced, "text")
    similarity_df: DataFrame = _pairs_df_model(embeddings_text)

    similarity_df_above_threshold: DataFrame = similarity_df[similarity_df["score"] > SIMILARITY_THRESHOLD]

    similarity_df_json: List[Dict[str, str]] = _groups_df(similarity_df_above_threshold, df_reduced)

    return similarity_df_json


def main() -> None:
    try:
        db: DataBase = DataBase()
        attractions: List[
            Dict[str, str]
        ] = db._fetch_attraction_ids_map()  # Needs some tweaking
        similarity_updates: List[Dict[str, str]] = compute_similarity_groups(
            attractions
        )
        queue: SQSQueue = SQSQueue(QUEUE_URL)
        queue.send_messages(similarity_updates, "similarity", 50)
        response = {
            "statusCode": 200,
            "body": {"message": "updates  were successfully sent to queue"},
        }
    except Exception as e:
        response: Dict[str, Any] = {
            "statusCode": 404,
            "body": {"message": "Unable to compute  similarity data", "details": e},
        }
    return response
