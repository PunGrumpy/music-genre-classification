import os
import sys
import time
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from src.pkg.CSVLogger import CSVLogger
from src.model.ModelBERT import ModelBERT
from src.MusicGenreDataset import MusicGenreDatasetWithPreprocess, _batch_to_tensor


class ModelTrainer:
    def __init__(self, settings, emb_model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load = settings["load_checkpoint"]
        self.run_mode = settings["mode"]
        if self.load != "" and self.run_mode == "train":
            checkpoint = self.get_checkpoint(settings["load_checkpoint"])
            self.output_dir = os.path.abspath("/".join(self.load.split("/")[:-2]))
            self.settings = checkpoint["settings"]
        else:
            self.settings = settings
            self.output_dir = settings["output_dir"]
            self.output_dir = os.path.join(
                self.output_dir,
                f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}_{settings['run_name']}",
            )

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        self.embedding_model_name = settings["embedding_settings"]["embedding_model"]
        if isinstance(emb_model, type(None)):
            if self.embedding_model_name != "distilbert-base-uncased":
                print(
                    f"Embedding model {self.embedding_model_name} not found. Using default distilbert-base-uncased"
                )
            elif self.embedding_model_name == "distilbert-base-uncased":
                print(f"Embedding model {self.embedding_model_name} is loading...")
                self.emb_model = AutoTokenizer.from_pretrained(
                    self.embedding_model_name
                )
        else:
            print(
                f"Embedding model {self.embedding_model_name} is loaded from parameter"
            )
            self.emb_model = emb_model

        # Import data
        print(f"Importing data...")
        self.data_train = MusicGenreDatasetWithPreprocess(
            path_data=settings["data_settings"]["train_data"],
            emb_model=self.emb_model,
            emb_type=settings["embedding_settings"]["embedding_type"],
            max_seq_len=settings["data_settings"]["max_seq_len"],
            input_type=settings["data_settings"]["input_type"],
            store_processed=settings["data_settings"]["store_processed"],
            output_dir=self.output_dir,
        )

        self.data_val = MusicGenreDatasetWithPreprocess(
            path_data=settings["data_settings"]["val_data"],
            emb_model=self.emb_model,
            emb_type=settings["embedding_settings"]["embedding_type"],
            max_seq_len=settings["data_settings"]["max_seq_len"],
            input_type=settings["data_settings"]["input_type"],
            store_processed=settings["data_settings"]["store_processed"],
            output_dir=self.output_dir,
        )

        self.data_test = MusicGenreDatasetWithPreprocess(
            path_data=settings["data_settings"]["test_data"],
            emb_model=self.emb_model,
            emb_type=settings["embedding_settings"]["embedding_type"],
            max_seq_len=settings["data_settings"]["max_seq_len"],
            input_type=settings["data_settings"]["input_type"],
            store_processed=settings["data_settings"]["store_processed"],
            output_dir=self.output_dir,
        )

        # Create dataloaders
        print(f"Creating dataloaders...")
        self.batch_size = self.settings["data_settings"]["batch_size"]
        self.dataloader_train = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_batch_to_tensor,
            drop_last=True,
        )

        self.dataloader_val = DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_batch_to_tensor,
            drop_last=True,
        )

        self.dataloader_test = DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_batch_to_tensor,
            drop_last=True,
        )

        # Create model
        print("Creating model...")
        if settings["embedding_settings"]["embedding_type"] == "bert":
            self.model = ModelBERT(
                lstm_layers=settings["model_settings"]["lstm_layers"],
                hidden_dim=settings["model_settings"]["hidden_dim"],
                target_size=settings["model_settings"]["target_size"],
                dropout_prob=settings["model_settings"]["dropout_prob"],
                device=self.device,
                bert_pretrained=settings["embedding_settings"]["embedding_model"],
                seq_len=settings["data_settings"]["max_seq_len"],
                train_bert=settings["model_settings"]["train_bert"],
            )
        self.model = self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.settings["train_settings"]["learning_rate"]
        )

        # Train settings
        if self.load != "" and self.run_mode == "eval":
            self.eval(data_loader=self.dataloader_test)
        elif self.load != "" and self.run_mode == "train":
            print(f"Model is loading... {self.load}")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.epochs = settings["train_settings"]["epochs"]
            self.batch_counter = 0
            self.epoch_counter = checkpoint["epoch"]
            self.grad_clip = settings["train_settings"]["grad_clip"]
            self.valid_loss_min = checkpoint["valid_loss_min"]
            self.valid_acc_max = checkpoint["valid_acc_max"]
            self.metrics = checkpoint["metrics"]
            self.csv_logger = CSVLogger(self.metrics, self.output_dir, self.batch_size)

            for index in range(len(self.metrics["train_loss"])):
                self.csv_logger.log(
                    {
                        "train_loss": self.metrics["train_loss"][index],
                        "val_loss": self.metrics["val_loss"][index],
                        "train_acc": self.metrics["train_acc"][index],
                        "val_acc": self.metrics["val_acc"][index],
                    }
                )
        elif self.load == "":
            self.epochs = settings["train_settings"]["epochs"]
            self.batch_counter = 0
            self.epoch_counter = 0
            self.grad_clip = settings["train_settings"]["grad_clip"]
            self.valid_loss_min = np.Inf  # Set initial minimum loss to infinity
            self.valid_acc_max = (
                np.NINF
            )  # Set initial maximum accuracy to negative infinity
            self.metrics = {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": [],
            }
            self.csv_logger = CSVLogger(self.metrics.keys())
            with open(os.path.join(self.output_dir, "settings.json"), "w") as f:
                json.dump(settings, f, indent=4)
        else:
            sys.exit(
                f"Load and mode settings are not compatible {self.load} {self.run_mode}"
            )

    def train(self):
        print_every = 100
        self.model.switch_train()

        while self.epoch_counter < self.epochs:
            self.epoch_counter += 1
            epoch = self.epoch_counter

            self.batch_counter = 0
            training_loss = 0.0
            training_accuracy = 0.0

            ep_start = time.time()
            temp_time = time.time()

            hidden = self.model.init_hidden(self.batch_size)

            for inputs, labels in self.dataloader_train:
                self.batch_counter += 1

                hidden = tuple([each.data for each in hidden])
                inputs = inputs.clone().detach().to(self.device)
                labels = labels.clone().detach().to(self.device)

                self.optimizer.zero_grad()  # Clear the gradients
                output, hidden = self.model.forward(inputs, hidden)

                predictions = torch.argmax(output, dim=1)[1]
                loss = self.loss_fn(output, labels)

                training_loss += loss.item()
                training_accuracy += (
                    torch.sum(predictions == labels) / labels.shape[0]
                )  # Calculate accuracy

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                if self.batch_size % print_every == 0 or self.batch_counter == 1:

                    print(
                        f"🏃‍♂️ Training...",
                        f"Epoch: {epoch}/{self.epochs}...",
                        f"Step: {self.batch_counter}...",
                        f"Loss: {training_loss / self.batch_counter:.6f}...",
                        f"Accuracy: {training_accuracy / self.batch_counter:.6f}...",
                        f"Time: {time.time() - temp_time:.6f}",
                    )
                    temp_time = time.time()

                if self.batch_counter == len(self.dataloader_train) - 1:
                    val_loss, val_acc = self.eval(self.dataloader_val)

                    row = {
                        "train_loss": training_loss / len(self.dataloader_train),
                        "val_loss": val_loss,
                        "train_acc": training_accuracy / len(self.dataloader_train),
                        "val_acc": val_acc,
                    }
                    self._log_metrics(row)

                    print(
                        f"🏋️ Evaluation...",
                        f"Epoch: {epoch}/{self.epochs}...",
                        f"Training Loss: {training_loss / len(self.dataloader_train):.6f}...",
                        f"Validation Loss: {val_loss:.6f}...",
                        f"Training Accuracy: {training_accuracy / len(self.dataloader_train):.6f}...",
                        f"Validation Accuracy: {val_acc:.6f}...",
                        f"Time: {time.time() - ep_start:.6f}",
                    )
                    temp_time = time.time()

                    # Save model
                    if self.valid_acc_max < row["val_acc"]:
                        self.valid_acc_max = row["val_acc"]

                    if self.valid_loss_min > row["val_loss"]:
                        self.valid_loss_min = row["val_loss"]
                        self.save_model(fname="min_loss_epoch")

                    self.save_model(fname="last_epoch")

                    ep_end_time = time.time()
                    print(f"Epoch time: {ep_end_time - ep_start:.6f}")

    def _log_metrics(self, row):
        self.metrics["train_loss"].append(row["train_loss"])
        self.metrics["train_acc"].append(row["train_acc"])
        self.metrics["val_loss"].append(row["val_loss"])
        self.metrics["val_acc"].append(row["val_acc"])
        self.csv_logger.log(row)

    def eval(self, data_loader=None):
        test_loss = 0.0
        test_accuracy = 0.0
        test_hidden = self.model.init_hidden(self.batch_size)

        self.model.eval()
        data_loader = (
            self.dataloader_test if isinstance(data_loader, type(None)) else data_loader
        )
        for inputs, labels in data_loader:
            test_hidden = tuple([each.data for each in test_hidden])

            inputs = inputs.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)
            output, test_hidden = self.model(inputs, test_hidden)

            predictions = torch.argmax(output, dim=1)[1]
            loss = self.loss_fn(output, labels)

            test_loss += loss.item()
            test_accuracy += torch.sum(predictions == labels) / labels.shape[0]

        self.model.switch_train()

        loss = test_loss / len(data_loader)
        accuracy = test_accuracy / len(data_loader)
        return loss, accuracy

    def prediction(self, data_loader=None):
        all_predictions = []
        all_labels = []
        test_hidden = self.model.init_hidden(self.batch_size)

        self.model.eval()
        data_loader = (
            self.dataloader_test if isinstance(data_loader, type(None)) else data_loader
        )

        for inputs, labels in data_loader:
            test_hidden = tuple([each.data for each in test_hidden])

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output, test_hidden = self.model(inputs, test_hidden)
            predictions = torch.argmax(output, dim=1)[1]

            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        self.model.switch_train()
        return np.array(all_predictions), np.array(all_labels)

    def save_model(self, fname="last_epoch"):
        state = {
            "settings": self.settings,
            "epoch": self.epoch_counter,
            "metrics": self.metrics,
            "valid_acc_max": self.valid_acc_max,
            "valid_loss_min": self.valid_loss_min,
            "optimizer": self.optimizer.state_dict(),
            "state_dict": self.model.state_dict(),
        }
        output_dir = os.path.join(self.output_dir, f"checkpoints")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        path = os.path.join(output_dir, f"{fname}.checkpoint")
        torch.save(state, path)

    def load_model(self, checkpoint_path):
        checkpoint = self.get_checkpoint(checkpoint_path)
        print(f"Model is loading... {checkpoint_path}")
        self.output_dir = os.path.abspath("/".join(checkpoint_path.split("/")[:-2]))
        self.settings = checkpoint["settings"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.epoch_counter = checkpoint["epoch"]
        self.valid_acc_max = checkpoint["valid_acc_max"]
        self.valid_loss_min = checkpoint["valid_loss_min"]
        self.metrics = checkpoint["metrics"]

    def get_checkpoint(self, checkpoint_path):
        return torch.load(checkpoint_path, map_location=self.device)
