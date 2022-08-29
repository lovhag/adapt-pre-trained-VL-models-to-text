import datasets
import os

LXMERT_DATA_PATH = "/data/lxmert/train_mlm.jsonl"
SAVE_DATA_PATH = "/data/lxmert/finetune"
TRAIN_SIZE = 9500
VAL_SIZE = 3300

#datasets.disable_caching()
dataset = datasets.load_dataset('json', data_files=LXMERT_DATA_PATH, split="train", keep_in_memory=True)
splitted_dataset = dataset.train_test_split(test_size=VAL_SIZE, train_size=TRAIN_SIZE, seed=42, shuffle=True, keep_in_memory=True)
splitted_dataset["train"].to_json(os.path.join(SAVE_DATA_PATH, "train.jsonl"), force_ascii=True)
splitted_dataset["test"].to_json(os.path.join(SAVE_DATA_PATH, "val.jsonl"), force_ascii=True)