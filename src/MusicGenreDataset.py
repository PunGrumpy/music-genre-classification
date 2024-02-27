import os
import spacy
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


class MusicGenreDataset(Dataset):
    def __init__(self, path_data, max_seq_length=250):
        self.max_seq_length = max_seq_length
        self.X, self.y_id, self.y_label = self._load_data(path_data)

    def __len__(self):
        return len(self.X)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        _X = np.zeros(self.max_seq_length, dtype=np.int64)
        item = self.X[index]
        max_index = (
            self.max_seq_length if self.max_seq_length < len(item) else len(item)
        )
        _X[:max_index] = np.int64(item[:max_index]) + 1
        return (index, _X, self.y_id[index], self.y_label[index])

    def _load_data(self, path_data):
        data = pd.read_csv(path_data, index_col=0)
        X_lyrics = data["lyrics"].values
        X_audio = data[AUDIO_FEATURES].values
        X = np.concatenate([X_lyrics.reshape(-1, 1), X_audio], axis=1)

        y_id = data["playlist_genre_id"].values
        y_label = data["playlist_genre"].values

        return X, y_id, y_label


def batch_to_tensor(batch):
    index, _X, target_id, target_label = zip(*batch)
    return torch.LongTensor(_X), torch.LongTensor(target_id)


class MusicGenreDatasetWithPreprocess(Dataset):
    def __init__(
        self,
        path_data,
        embedding_model,
        embedding_type,
        max_seq_length=250,
        input_type="index",
        store_preprocess=False,
        output_dir="",
    ):
        self.input_type = input_type
        self.emb_model = embedding_model
        self.emb_type = embedding_type
        self.store_preprocess = store_preprocess
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir

        if self.input_type == "unclean":
            self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

        if self.emb_type == "bert":
            self.func_w2i = self._bert_w2id
        elif self.emb_type == "gensim":
            self.func_w2i = self._gensim_w2id
        else:
            os.error(f"Embedding type {self.emb_type} not supported")

        self.X, self.y_id, self.y_label = self._load_data(path_data)

    def __len__(self):
        return len(self.X)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        _X = np.zeros(self.max_seq_len, dtype=np.int64)
        item = self.X[index]

        if not self.store_preprocess:
            if self.input_type == "unclean":
                item = self.preprocess(self.nlp, " ".join(item))
                item = self.func_w2i(item)
            elif self.input_type == "clean":
                item = self.func_w2i(item)
            elif self.input_type == "index":
                pass
            else:
                os.error(f"Input type {self.input_type} not supported")

        max_index = (
            self.max_seq_length if self.max_seq_length < len(item) else len(item)
        )
        const = 1 if self.emb_type == "gensim" else 0
        _X[:max_index] = np.int64(item[:max_index]) + const
        return (index, _X, self.y_id[index], self.y_label[index])

    def _load_data(self, path_data):
        data = pd.read_csv(path_data, index_col=0)
        if self.store_preprocess and self.input_type != "index":
            print(f"Storing preprocessed data to {self.output_dir}")

            if self.input_type == "unclean":
                X_lyrics = [
                    self.preprocess(self.nlp, " ".join(item))
                    for item in data["lyrics"].values
                ]
                X_lyrics = [self.func_w2i(item) for item in X_lyrics]
            elif self.input_type == "clean":
                pass
