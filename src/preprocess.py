import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold


def prepare_skf_dataset(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
    df['matches'] = df['label_group'].map(tmp)
    df['matches'] = df['matches'].apply(lambda x: ' '.join(x))

    encoder = LabelEncoder()
    df['label_group'] = encoder.fit_transform(df['label_group'])
    
    df['fold'] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df['title'], df['label_group'], groups=df['label_group'])):
        df.loc[valid_idx, 'fold'] = fold
    
    return df


def prepare_gkf_dataset(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
    df['matches'] = df['label_group'].map(tmp)
    df['matches'] = df['matches'].apply(lambda x: ' '.join(x))

    encoder = LabelEncoder()
    df['label_group'] = encoder.fit_transform(df['label_group'])
    
    df['fold'] = -1
    skf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df['title'], df['label_group'], groups=df['label_group'])):
        df.loc[valid_idx, 'fold'] = fold
    
    return df
