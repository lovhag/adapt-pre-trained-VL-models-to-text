import datasets
import os

WIKIPEDIA_PATH = "/data/wikipedia/train_subset.jsonl"
SAVE_DATA_PATH = "/data/wikipedia/finetune"
TRAIN_SIZE = 4600 #scaled to match LXMERT in number of tokens
VAL_SIZE = 1600

#datasets.disable_caching()
dataset = datasets.load_dataset('json', data_files=WIKIPEDIA_PATH, split="train", keep_in_memory=True)
splitted_dataset = dataset.train_test_split(test_size=VAL_SIZE, train_size=TRAIN_SIZE, seed=42, shuffle=True, keep_in_memory=True)
splitted_dataset["train"].to_json(os.path.join(SAVE_DATA_PATH, "train.jsonl"), force_ascii=True)
splitted_dataset["test"].to_json(os.path.join(SAVE_DATA_PATH, "val.jsonl"), force_ascii=True)