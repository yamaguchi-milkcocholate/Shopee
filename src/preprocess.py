import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
    df['matches'] = df['label_group'].map(tmp)
    df['matches'] = df['matches'].apply(lambda x: ' '.join(x))

    encoder = LabelEncoder()
    df['label_group'] = encoder.fit_transform(df['label_group'])
    df['fold'] = -1

    return df


def prepare_skf_split_dataset(df: pd.DataFrame, seed: int, ratio: float):
    df = _prepare_dataset(df=df)

    train_idx, valid_idx = train_test_split(df.index, shuffle=True, random_state=seed, stratify=df['label_group'].values, test_size=ratio)
    df.loc[train_idx, 'fold'] = 0
    df.loc[valid_idx, 'fold'] = 1

    return df

def prepare_skf_dataset(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    df = _prepare_dataset(df=df)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df['title'], df['label_group'], groups=df['label_group'])):
        df.loc[valid_idx, 'fold'] = fold
    
    return df


def prepare_gkf_dataset(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    df = _prepare_dataset(df=df)

    skf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df['title'], df['label_group'], groups=df['label_group'])):
        df.loc[valid_idx, 'fold'] = fold
    
    return df
