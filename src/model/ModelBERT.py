import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel


class ModelBERT(nn.Module):
    def __init__(
        self,
        lstm_layers,
        hidden_dim,
        target_size,
        dropout_prob,
        device,
        seq_len=250,
        bert_pretrained="distilbert-base-uncased",
        train_bert=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.device = device

        self.bert_pretrained = bert_pretrained
        self.train_bert = train_bert
        self.bert_model = AutoModel.from_pretrained(self.bert_pretrained)
        self.bert_emb_dim = self.bert_model.config.hidden_size  # 768

        self.lstm = nn.LSTM(
            input_size=self.bert_emb_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            dropout=dropout_prob,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_dim * seq_len, 1024)
        self.hidden2tag = nn.Linear(hidden_dim * seq_len, target_size)
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence, hidden):
        with torch.no_grad():  # Freeze BERT layers
            embedded = self.bert_model(sentence)[0]

        x, (hidden, cell) = self.lstm(embedded, hidden)
        x = x.contiguous().view(x.shape[0], -1)  # Flatten the output
        x = self.dropout(x)
        # x = self.linear(x)
        x = self.hidden2tag(x)
        # x = self.sigmoid(x)
        x = self.log_softmax(x)

        return x, (hidden, cell)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.lstm_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(self.device),
            weight.new(self.lstm_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(self.device),
        )
        return hidden

    def switch_train(self):
        self.train()
        bert_model = next(self.children())
        bert_model.train(self.train_bert)
