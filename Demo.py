import torch
import numpy as np
import gradio as gr
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

REPO_NAME = "PunGrumpy/music-genre-classification"

GENRE = {"edm": 0, "r&b": 1, "rap": 2, "rock": 3, "pop": 4}
AUDIO_FEATURES = {
    "acousticness": 0,
    "danceability": 0,
    "energy": 0,
    "instrumentalness": 0,
    "key": 0,
    "liveness": 0,
    "loudness": 0,
    "mode": 0,
    "speechiness": 0,
    "tempo": 0,
    "valence": 0,
}


class LyricsAudioModelInference:
    def __init__(self, model_name, num_labels=5):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_labels = num_labels
        self.classifier = nn.Linear(
            self.model.config.hidden_size + len(AUDIO_FEATURES), num_labels
        )

    def predict_genre(self, lyrics: str, *audio_features) -> dict:
        input_lyrics = self.tokenizer(
            lyrics, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        outputs = self.model(**input_lyrics)
        lyrics_embedding = outputs.last_hidden_state.mean(dim=1)

        if audio_features is not None:
            audio_features = list(audio_features)

            input_features = torch.cat(
                [lyrics_embedding, torch.tensor(audio_features).float().unsqueeze(0)],
                dim=1,
            )
        else:
            input_features = lyrics_embedding

        logits = self.classifier(input_features)
        probs = F.softmax(logits, dim=1)
        print(logits)
        print(probs)

        top3_genres = torch.topk(probs, k=3, dim=1)
        result = {}
        for i in range(3):
            genre_idx = top3_genres.indices[0][i].item()
            genre_prob = top3_genres.values[0][i].item()
            genre_label = [key for key, value in GENRE.items() if value == genre_idx][0]
            result[genre_label] = genre_prob

        return result


if __name__ == "__main__":
    iface = gr.Interface(
        fn=LyricsAudioModelInference(model_name=REPO_NAME).predict_genre,
        inputs=[
            gr.Textbox(
                lines=20,
                placeholder="Enter lyrics here...",
                label="Lyrics",
            ),
            gr.Slider(
                minimum=0,
                maximum=1,
                label="Acousticness",
                step=0.01,
            ),
            gr.Slider(
                minimum=0,
                maximum=1,
                label="Danceability",
                step=0.01,
            ),
            gr.Slider(minimum=0, maximum=1, label="Energy", step=0.01),
            gr.Slider(
                minimum=0,
                maximum=1,
                label="Instrumentalness",
                step=0.01,
            ),
            gr.Slider(minimum=0, maximum=11, label="Key", step=1),
            gr.Slider(minimum=0, maximum=1, label="Liveness", step=0.01),
            gr.Slider(minimum=-60, maximum=0, label="Loudness", step=1),
            gr.Slider(minimum=0, maximum=1, label="Mode", step=1),
            gr.Slider(minimum=0, maximum=1, label="Speechiness", step=0.01),
            gr.Slider(minimum=0, maximum=200, label="Tempo", step=1),
            gr.Slider(minimum=0, maximum=1, label="Valence", step=0.01),
        ],
        outputs=gr.Label(
            num_top_classes=3,
            label="Top 3 Predicted Genres",
        ),
        title="Music Genre Classifier",
        description="This model predicts the genre of a song based on its lyrics and audio features.",
        examples=[
            [
                "When the sun is rising over streets so barren...",
                0.7050,
                0.420,
                0.247,
                0.00349,
                2,
                0.1270,
                -13.370,
                0,
                0.0360,
                88.071,
                0.138,
            ],
        ],
    )

    iface.launch(debug=True)
