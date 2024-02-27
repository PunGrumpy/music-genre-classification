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


class InputType:
    UNCLEAN = "unclean"
    CLEAN = "clean"
    INDEX = "index"


class MusicGenreDataset(Dataset):
    def __init__(
        self,
        path_data: str,
        embedder_model: any,
        embedder_type: str,
        max_seq_length: int = 250,
    ):
        self.embedder_model = embedder_model
        self.embedder_type = embedder_type
        self.max_seq_length = max_seq_length
        self.X, self.X_audio, self.y_label, self.y_id = self._load_data(path_data)

    def __len__(self) -> int:
        return len(self.X)

    def __iter__(self) -> any:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index) -> tuple:
        item = self.X[index]

        if self.embedder_type == "bert":
            input = self.embedder_model.encode_plus(
                item,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            ids = input["input_ids"]
            mask = input["attention_mask"]
            token_type_ids = input["token_type_ids"]
        elif self.embedder_type == "gensim":
            ids = [int(word) for word in item.split()]
            mask = np.ones(self.max_seq_length, dtype=np.int64)
            token_type_ids = np.zeros(self.max_seq_length, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported embedder type: {self.embedder_type}")

        return (
            torch.tensor(ids, dtype=torch.int64),
            torch.tensor(mask, dtype=torch.int64),
            torch.tensor(token_type_ids, dtype=torch.int64),
            torch.tensor(self.y_id[index], dtype=torch.int64),
            torch.tensor(self.X_audio[index], dtype=torch.float32),
        )

    def _load_data(self, path_data: str) -> pd.DataFrame:
        data = pd.read_csv(path_data, index_col=0)
        X, X_audio, y_label, y_id = self._load_raw_data(data)
        return X, X_audio, y_label, y_id

    def _load_raw_data(self, data: pd.DataFrame) -> tuple:
        if self.embedder_type == "gensim":
            X = [lyrics.split() for lyrics in data["lyrics"].values]
        else:
            X = data["lyrics"].values
        return (
            X,
            data[AUDIO_FEATURES],
            data["playlist_genre"],
            data["playlist_genre_id"],
        )


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
        self.input_type = input_type
        self.embedder_model = embedder_model
        self.embedder_type = embedder_type
        self.max_seq_length = max_seq_length
        self.store_processed = store_processed
        self.output_dir = os.path.abspath(output_dir)
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        if self.embedder_type == "bert":
            self.func_w2v = self._bert_w2v
        elif self.embedder_type == "gensim":
            self.func_w2v = self._gensim_w2v
        else:
            raise ValueError(f"Unsupported embedder type: {self.embedder_type}")

        self.X, self.X_audio, self.y_label, self.y_id = self._load_data(path_data)

    def __len__(self) -> int:
        return len(self.X)

    def __iter__(self) -> any:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index) -> tuple:
        item = self.X[index]

        if self.embedder_type == "bert":
            input = self.embedder_model.encode_plus(
                item,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            ids = input["input_ids"]
            mask = input["attention_mask"]
            token_type_ids = input["token_type_ids"]
        elif self.embedder_type == "gensim":
            ids = [int(word) for word in item.split()]
            mask = np.ones(self.max_seq_length, dtype=np.int64)
            token_type_ids = np.zeros(self.max_seq_length, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported embedder type: {self.embedder_type}")

        return (
            torch.tensor(ids, dtype=torch.int64),
            torch.tensor(mask, dtype=torch.int64),
            torch.tensor(token_type_ids, dtype=torch.int64),
            torch.tensor(self.y_id[index], dtype=torch.int64),
            torch.tensor(self.X_audio[index], dtype=torch.float32),
        )

    def _process_item(self, item) -> list:
        if self.input_type == InputType.UNCLEAN:
            item = self.preprocess(self.nlp, " ".join(item))
            item = self.func_w2v(item)
        elif self.input_type == InputType.CLEAN:
            item = self.func_w2v(item)
        elif self.input_type == InputType.INDEX:
            pass
        else:
            raise ValueError(f"Unsupported input type: {self.input_type}")

        return item

    def _load_data(self, path_data: str) -> pd.DataFrame:
        data = pd.read_csv(path_data, index_col=0)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if self.store_processed and self.input_type != InputType.INDEX:
            X, X_audio, y_label, y_id = self._process_data(data)
            path_processed = os.path.join(
                self.output_dir,
                f"ids_{datetime.now().strftime('%y-%m-%d_%H%M%S')}.csv.zip",
            )

            tmp = pd.DataFrame(
                {
                    "lyrics": [" ".join(map(str, lyrics)) for lyrics in X.values],
                    "_lyrics": data["lyrics"].values,
                    **{feature: data[feature].values for feature in AUDIO_FEATURES},
                    "playlist_genre": data["playlist_genre"].values,
                    "playlist_genre_id": data["playlist_genre_id"].values,
                }
            )
            tmp.to_csv(path_processed, compression="zip")
        else:
            X, X_audio, y_label, y_id = self._load_raw_data(data)

        return X, X_audio, y_label, y_id

    def _process_data(self, data: pd.DataFrame) -> tuple:
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
            raise ValueError(f"Unsupported input type: {self.input_type}")

        return (
            X,
            data[AUDIO_FEATURES],
            data["playlist_genre"],
            data["playlist_genre_id"],
        )

    def _load_raw_data(self, data: pd.DataFrame) -> tuple:
        if self.input_type == InputType.INDEX:
            X = [
                [int(word) for word in lyrics.split()]
                for lyrics in data["lyrics"].values
            ]
        else:
            X = [[word for word in lyrics.split()] for lyrics in data["lyrics"].values]

        return (
            X,
            data[AUDIO_FEATURES],
            data["playlist_genre"],
            data["playlist_genre_id"],
        )

    def _bert_w2v(self, tokens: list) -> list:
        return [self.embedder_model.encode(token) for token in tokens]

    def _gensim_w2v(self, tokens: list) -> list:
        return [
            self.embedder_model[token]
            for token in tokens
            if token in self.embedder_model
        ]

    @staticmethod
    def preprocess(nlp, text: str) -> list:
        """
        Perform text preprocessing using spaCy.

        Parameters:
        - nlp (spacy.lang.en.English): SpaCy language model.
        - text (str): Input text.

        Returns:
        - list: Processed list of tokens.
        """
        doc = nlp(text)
        out = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and token.lemma_.isalpha()
        ]
        return out
