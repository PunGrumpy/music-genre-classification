BERT_SETTINGS = {
    "run_name": "BERT Model",
    "output_dir": "./output/BERT",
    "load_checkpoint": "",
    "mode": "train",  # train or eval
    "embedding_settings": {
        "embedding_type": "bert",
        "embedding_model": "distilbert-base-uncased",
    },
    "data_settings": {
        "train_data": "./data/BERT/bert_ids_train.csv.zip",
        "val_data": "./data/BERT/bert_ids_val.csv.zip",
        "test_data": "./data/BERT/bert_ids_test.csv.zip",
        "max_seq_len": 500,
        "input_type": "index",
        "store_processed": False,
        "batch_size": 96,
    },
    "model_settings": {
        "lstm_layers": 2,
        "hidden_dim": 128,
        "target_size": 5,  # Rap, Rock, Pop, R&B, and EDM
        "dropout_prob": 0.2,
        "train_bert": False,
    },
    "train_settings": {
        "epochs": 1,
        "learning_rate": 5e-6,
        "grad_clip": 5,  # Gradient clipping
    },
}
