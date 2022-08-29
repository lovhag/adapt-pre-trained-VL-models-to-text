import argparse
import datasets
import os

from spacy.lang.en import English
from unidecode import unidecode

def get_args():
    parser = argparse.ArgumentParser(description="Extracts a subset of a corpus")
    parser.add_argument("--dataset-name", type=str, help="What dataset to process and load")
    parser.add_argument("--revision", type=str, default=None, help="What revision from the dataset to process and load")
    
    parser.add_argument("--save-dir", type=str, help="What directory to save dataset and processed subparts to")
    parser.add_argument("--cache-dir", type=str, help="What directory to use for caching")
    parser.add_argument("--nbr-train-samples", type=int, help="Number of samples to include in train set")
    parser.add_argument("--nbr-val-samples", type=int, help="Number of samples to include in validation set")

    return parser.parse_args()

def get_split_pos(text):
    period_pos = text.find(".")
    period_pos = period_pos if not period_pos ==-1 else float("inf")
    
    exclamation_pos = text.find("!")
    exclamation_pos = exclamation_pos if not exclamation_pos ==-1 else float("inf")
    
    newline_pos = text.find("\n")
    newline_pos = newline_pos-1 if not newline_pos ==-1 else float("inf") # do not want to include the newline character
    
    end_marker_pos = min([period_pos, exclamation_pos, newline_pos])
    
    # if standard sentence end comes first, return this sentence
    if not isinf(end_marker_pos):
        return end_marker_pos+1
    # if no typical end exists, indicate this
    else:
        return -1
        
def split_to_sentences(examples, sentencizer):
    return_examples = []
    
    for text in examples["text"]:
        for text_part in text.split("\n"):
            for sentence in sentencizer(text_part.strip()).sents:
                sentence = str(sentence).strip()
                if sentence.count(" ") > 0:
                    return_examples += [sentence]
        
    return {"text": return_examples}

def transform_to_ascii(example):
    return {"text": unidecode(example["text"])}

def process_dataset_parts(dataset_name, 
                          revision,
                          save_dir, 
                          cache_dir, 
                          process_splits=["train[:10%]", "train[10%:20%]", "train[20%:30%]", "train[30%:40%]", "train[40%:50%]", "train[50%:60%]", "train[60%:70%]", "train[70%:80%]", "train[80%:90%]", "train[90%:]"]):
    # process the dataset by splitting it into sentences and turning to ascii
    # do it in parts due to memory issues
    nlp = English()
    nlp.add_pipe("sentencizer", config={"punct_chars": [".", "!", "。"]})
    
    for process_split in process_splits:
        dataset = datasets.load_dataset(dataset_name, revision, cache_dir=cache_dir, split=process_split)
        #print(dataset.info)

        chunked_dataset = dataset.map(lambda x: split_to_sentences(x, nlp), batched=True, batch_size=10, writer_batch_size=10, remove_columns=dataset.column_names, cache_file_name=os.path.join(cache_dir, "mappings", f"to_sentences_{process_split}.arrow"), num_proc=4, keep_in_memory=False)
        chunked_dataset = chunked_dataset.map(transform_to_ascii, batched=False, writer_batch_size=10, cache_file_name=os.path.join(cache_dir, "mappings", f"to_ascii_{process_split}.arrow"), num_proc=4, keep_in_memory=False)
        filename = os.path.join(save_dir, f"sentences_{process_split}.jsonl")
        filename = filename.replace("[", "").replace("]", "")
        chunked_dataset.to_json(filename, force_ascii=True)

def concat_dataset_parts(dataset_save_path,
                         save_dir,
                         load_splits=["train:10%", "train10%:20%", "train20%:30%", "train30%:40%", "train40%:50%", "train50%:60%", "train60%:70%", "train70%:80%", "train80%:90%", "train90%:"]):
    # concatenate processed parts into one large
    dataset_parts = []
    for load_split in load_splits:
        dataset_parts.append(datasets.load_dataset('json', data_files=os.path.join(save_dir, f"sentences_{load_split}.jsonl"), split="train"))
        
    chunked_dataset = datasets.concatenate_datasets(dataset_parts)
    chunked_dataset.to_json(dataset_save_path, force_ascii=True)
    
    return chunked_dataset

def main(args):
    main_dataset_path = os.path.join(args.save_dir, "sentences.jsonl")
    if not os.path.exists(main_dataset_path):
        process_dataset_parts(args.dataset_name,
                              revision=args.revision,
                              save_dir=args.save_dir,
                              cache_dir=args.cache_dir,
                              process_splits=["train[:10%]", "train[10%:20%]", "train[20%:30%]", "train[30%:40%]", "train[40%:50%]", "train[50%:60%]", "train[60%:70%]", "train[70%:80%]", "train[80%:90%]", "train[90%:]"])
        dataset = concat_dataset_parts(dataset_save_path=main_dataset_path,
                                       save_dir=args.save_dir,
                                       load_splits=["train:10%", "train10%:20%", "train20%:30%", "train30%:40%", "train40%:50%", "train50%:60%", "train60%:70%", "train70%:80%", "train80%:90%", "train90%:"])
    else:
        # load processed data
        dataset = datasets.load_dataset('json', data_files=main_dataset_path, split="train")
        
    print(f"Number of sentences in {args.dataset_name}: {len(dataset)}")
    
    # create train/test subsets
    splitted_dataset = dataset.train_test_split(test_size=args.nbr_val_samples, train_size=args.nbr_train_samples, seed=42, shuffle=True)

    splitted_dataset["train"].to_json(os.path.join(args.save_dir, "train_subset.jsonl"))
    splitted_dataset["test"].to_json(os.path.join(args.save_dir, "test_subset.jsonl"))

if __name__ == "__main__":
    args = get_args()
    main(args)