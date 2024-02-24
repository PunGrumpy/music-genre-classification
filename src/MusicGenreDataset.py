import os
import sys
import spacy
import torch
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from datetime import datetime
import gensim.downloader as api
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def _batch_to_tensor(batch):
    index, _X, target_id, target_label = zip(*batch)
    return (
        torch.LongTensor(np.array(_X)),
        torch.LongTensor(np.array(target_id)).squeeze(),
    )


class MusicGenreDatasetWithPreprocess(Dataset):
    def __init__(
        self,
        path_data,
        emb_model,
        emb_type,
        max_seq_len=250,
        input_type="index",
        store_processed=False,
        output_dir="",
    ):
        self.input_type = input_type
        self.emb_model = emb_model
        self.emb_type = emb_type
        self.store_processed = store_processed
        self.max_seq_len = max_seq_len
        self.output_dir = os.path.abspath(output_dir)

        if self.input_type == "unclean":
            self.nlp = spacy.load(
                "en_core_web_sm",
                disable=[
                    "tagger",
                    "parser",
                    "ner",
                ],  # if disable "lemmatizer" it not work on tagging IDs on lyrics
            )

        if self.emb_type == "bert":
            self.func_w2i = self._bert_w2id
        else:
            sys.exit(f"Unrecognized embedding type: {self.emb_type}")

        # Lyrics, genre_id, genre, and acousticness
        self.X, self.y_id, self.y_label, self.acousticness = self._load_data(path_data)

    def _load_data(self, path_data):
        data = pd.read_csv(path_data, index_col=[0])

        if self.store_processed and self.input_type != "index":
            print(
                f"Store processed activated. \nPreprocessing data for {self.input_type} input..."
            )

            if self.input_type == "unclean":
                print("Preprocessing and tokenizing input data...")
                X = [self.preprocess(self.nlp, lyric) for lyric in data.lyrics.values]
                print("Convert tokenized to IDs...")
                X = [self.func_w2i(token_list) for token_list in X]
            elif self.input_type == "clean":
                X = [lyric.split() for lyric in data.lyrics.values]
                X = [self.func_w2i(token_list) for token_list in X]
            else:
                sys.exit(f"Unrecognized input type: {self.input_type}")

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            path_output = os.path.join(
                self.output_dir,
                f"ids_{datetime.now().strftime('%y-%m-%d_%H%M%S')}.csv.zip",
            )
            print(
                f"Preprocessed end. \nStoring processed data to {path_output.split('/')[-1]}"
            )

            tmp = pd.DataFrame(
                {
                    "lyrics": [" ".join([str(word) for word in lyric]) for lyric in X],
                    "_lyrics": data["lyrics"].values,
                    "playlist_genre": data["playlist_genre"].values,
                    "playlist_genre_id": data["playlist_genre_id"].values,
                    "acousticness": data["acousticness"].values,
                }
            )
            tmp.to_csv(path_output, compression="zip")

        else:
            if self.input_type == "index":
                X = [
                    [int(word) for word in lyric.split()]
                    for lyric in data["lyrics"].values
                ]
            else:
                X = [
                    [word for word in lyric.split()] for lyric in data["lyrics"].values
                ]

        y_label = data["playlist_genre"].values
        y_id = data["playlist_genre_id"].values
        acousticness = data["acousticness"].values

        return (X, y_id, y_label, acousticness)

    def __len__(self):
        return len(self.X)

    def __iter__(self):
        for index in range(len(self)):
            yield self(index)

    def __getitem__(self, index):
        _X = np.zeros(self.max_seq_len, dtype=np.int64)
        item = self.X[index]

        if not self.store_processed:
            if self.input_type == "unclean":
                item = self.preprocess(self.nlp, " ".join(item))
                item = self.func_w2i(item)
            elif self.input_type == "clean":
                item = self.func_w2i(item)
            elif self.input_type == "index":
                pass
            else:
                sys.exit(f"Unrecognized input type {self.input_type}")

        max_i = self.max_seq_len if self.max_seq_len < len(item) else len(item)
        const = 1 if self.emb_type == "gensim" else 0
        _X[:max_i] = np.int64(item[:max_i]) + const
        return (
            index,
            _X,
            self.y_id[index],
            self.y_label[index],
            # self.acousticness[index],
        )

    def _bert_w2id(self, tokens):
        ids = [self.emb_model.encode(token)[0] for token in tokens]
        return ids

    @staticmethod
    def preprocess(nlp, text):
        """returns a list of preprocessed tokens (tokens are strings/words)"""
        doc = nlp(text)
        out = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and token.lemma_.isalpha()
        ]
        return out
