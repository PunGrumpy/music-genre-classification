import os
import sys
import spacy
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset


class InputType:
    UNCLEAN = "unclean"
    CLEAN = "clean"
    INDEX = "index"


class MusicGenreDatasetWithPreprocess(Dataset):
    def __init__(
        self,
        path_data: str,
        embedder_model: any,
        embedder_type: str,
        input_type: InputType,
        max_seq_length: int = 250,
        store_processed: bool = True,
        output_dir: str = "",
    ):
        """Music Genre Dataset with Preprocessing

        Args:
            path_data (str): Path to the dataset
            embedder_model (str): Model name for the embedder
            embedder_type (str): Type of embedder
            input_type (str): Type of input
            max_seq_length (int, optional): Maximum sequence length. Defaults to 250.
            store_processed (bool, optional): Store the processed data. Defaults to True.
            output_dir (str, optional): Output directory. Defaults to "".
        """
        self.input_type = input_type
        self.embedder_model = embedder_model
        self.embedder_type = embedder_type
        self.max_seq_length = max_seq_length
        self.store_processed = store_processed
        self.output_dir = os.path.abspath(output_dir)
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.genre_features = ["playlist_genre", "playlist_genre_id"]
        self.audio_features = [
            "danceability",
            "energy",
            "key",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        ]

        if self.embedder_type == "bert":
            self.func_w2v = self._bert_w2v
        else:
            sys.exit(f"Embedder type {self.embedder_type} not supported")

        self.X, self.y_label, self.y_id, self.y_audio = self._load_data(path_data)

    def __len__(self) -> int:
        """Length of the dataset"""
        return len(self.X)

    def __iter__(self):
        """Iterate over the dataset"""
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index) -> tuple:
        """Get item"""
        _X = np.zeros(self.max_seq_length, dtype=np.int64)
        item = self.X[index]

        if not self.store_processed:
            item = self._process_item(item)

        return (
            torch.tensor(item, dtype=torch.float32),
            torch.tensor(self.y_id[index], dtype=torch.int64),
            torch.tensor(self.y_audio[index], dtype=torch.float32),
        )

    def _process_item(self, item) -> list:
        """Process item based on input type"""
        if self.input_type == InputType.UNCLEAN:
            item = self.preprocess(self.nlp, " ".join(item))
            item = self.func_w2v(item)
        elif self.input_type == InputType.CLEAN:
            item = self.func_w2v(item)
        elif self.input_type == InputType.INDEX:
            pass
        else:
            sys.exit(f"Input type {self.input_type} not supported")

        return item

    def _load_data(self, path_data: str) -> pd.DataFrame:
        """Load and preprocess data"""
        data = pd.read_csv(path_data, index_col=0)

        if self.store_processed and self.input_type != InputType.INDEX:
            print(
                f"Store processed data is activated. Saving processed data to {self.output_dir.split('/')[-2]}/{self.output_dir.split('/')[-1]} ..."
            )
            X, y_label, y_id, y_audio = self._process_data(data)
            path_processed = os.path.join(
                self.output_dir,
                f"ids_{datetime.now().strftime('%y-%m-%d_%H%M%S')}.csv.zip",
            )
            print(f"Preprocessing done. Saving to {path_processed.split('/')[-1]} ...")

            tmp = pd.DataFrame(
                {
                    "lyrics": [" ".join(map(str, lyrics)) for lyrics in X.values],
                    "_lyrics": data["lyrics"].values,
                    **{
                        feature: data[feature].values for feature in self.genre_features
                    },
                    **{
                        feature: data[feature].values for feature in self.audio_features
                    },
                }
            )
            tmp.to_csv(path_processed, compression="zip")
        else:
            X, y_label, y_id, y_audio = self._load_raw_data(data)

        return X, y_label, y_id, y_audio

    def _process_data(self, data: pd.DataFrame) -> tuple:
        """Process data based on input type"""
        if self.input_type == InputType.UNCLEAN:
            X = [self.preprocess(self.nlp, lyrics) for lyrics in data["lyrics"].values]
            X = [self.func_w2v(tokens) for tokens in X]
            X = pd.DataFrame(X)
        elif self.input_type == InputType.CLEAN:
            X = [lyrics.split() for lyrics in data["lyrics"].values]
            X = [self.func_w2v(tokens) for tokens in X]
            X = pd.DataFrame(X)
        elif self.input_type == InputType.INDEX:
            X = [
                [int(word) for word in lyrics.split()]
                for lyrics in data["lyrics"].values
            ]
        else:
            sys.exit(f"Input type {self.input_type} not supported")

        return (
            X,
            data[self.genre_features],
            data["playlist_genre_id"],
            data[self.audio_features],
        )

    def _load_raw_data(self, data: pd.DataFrame) -> tuple:
        """Load raw data based on input type"""
        if self.input_type == InputType.INDEX:
            X = [
                [int(word) for word in lyrics.split()]
                for lyrics in data["lyrics"].values
            ]
        else:
            X = [[word for word in lyrics.split()] for lyrics in data["lyrics"].values]

        return (
            X,
            data[self.genre_features],
            data["playlist_genre_id"],
            data[self.audio_features],
        )

    def _bert_w2v(self, tokens: list) -> list:
        """BERT word embeddings"""
        return [self.embedder_model.encode(token) for token in tokens]

    @staticmethod
    def preprocess(nlp, text: str) -> list:
        """Preprocess text using spaCy.

        This method takes raw text as input and performs the following preprocessing steps:
        - Lemmatization: reducing words to their base or root form
        - Lowercasing: converting all characters to lowercase
        - Stop words removal: eliminating common words that usually don't contribute much to the meaning
        """
        doc = nlp(text)
        out = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and token.lemma_.isalpha()
        ]
        return out
