# General data

This folder contains the code to generate the Wikipedia and LXMERT datasets used in the project. Its purposes and contents can be described as follows:
* Process Wikipedia and LXMERT text data from the raw data files to create datasets for text-only training. The Wikipedia dataset is also adapted to have the same size as the LXMERT text data.
* Analyze the processed datasets in [analysis-ipynb](analysis.ipynb).
* Create subsets of the Wikipedia and LXMERT text datasets for finetuning.

## How to acqurie the datasets necessary for the project

The already derived and processed datasets can be found in the links below for each type. 

## How to generate the datasets necessary for the project

You can generate or download the necessary datasets by following the steps listed below.

### 1. Download the LXMERT text data

Already generated version available for download [here](https://huggingface.co/datasets/Lo/adapt-pre-trained-VL-models-to-text-data-LXMERT).

First, download the `mscoco_train.json`, `mscoco_nominival.json`, `vgnococo.json` and `mscoco_minival.json` data files as described in https://github.com/airsplay/lxmert#pre-training and put them under [/data/lxmert](/data/lxmert).

Then follow the instructions and run [/data/lxmert/generate_set.ipynb](/data/lxmert/generate_set.ipynb). After this, you should have the following files:
* `data/lxmert/train_mlm.jsonl`
* `data/lxmert/val_mlm.jsonl`

### 2. Download the Wikipedia data

Already generated version available for download [here](https://huggingface.co/datasets/Lo/adapt-pre-trained-VL-models-to-text-data-Wikipedia).

From https://huggingface.co/datasets/wikipedia

To get the data run the following code standing in the [/data/wikipedia](/data/wikipedia) folder:

```bash
python generate_subset.py --dataset-name "wikipedia" --revision "20200501.en" --save-dir "." --cache-dir "cache" --nbr-train-samples 4400000 --nbr-val-samples 100000
```

Number of samples are adapted to that we only want 59M tokens for training and 1.5M for testing (same size as LXMERT data). 

The following processing of the Wikipedia dataset is performed:
* Split on newline (\n) characters
* Split into sentences using SpaCy sentencizer
    * with "punct_chars": [".", "!", "。"]
* Remove examples that do not contain at least one whitespace
    * To remove single words
* Convert to ASCII format

After this, you should have the following files:
* `data/wikipedia/train_subset.jsonl` (~402MB)
* `data/wikipedia/test_subset.jsonl` (~9.1MB)

You can now analyse the datasets in [/data/analysis.ipynb](/data/analysis.ipynb) should you want to. The notebook analyzes the number of tokens per sample for the datasets. This then informs us about how many samples to include in the final dataset versions (such that they match LXMERT in size).

Results from analyzing 9M train samples and 0.21M test samples of each dataset:
* LXMERT train subset contains approximately 59M tokens (length: 8954401 samples), 6.6 words/sample
* Wikipedia train subset contains approximately 121M tokens (length: 9000000 samples), 13.4 words/sample

### 3. Create a finetuning set from LXMERT text data

Already generated version available for download [here](https://huggingface.co/datasets/Lo/adapt-pre-trained-VL-models-to-text-data-LXMERT-finetune).

Run `data/lxmert/generate_finetune_data.py`

After this, you should have the following files:
* `/data/lxmert/finetune/train.jsonl`
* `/data/lxmert/finetune/val.jsonl`

### 4. Create a finetuning set from Wikipedia data

Already generated version available for download [here](https://huggingface.co/datasets/Lo/adapt-pre-trained-VL-models-to-text-data-Wikipedia-finetune).

Run `data/wikipedia/generate_finetune_data.py`

After this, you should have the following files:
* `/data/wikipedia/finetune/train.jsonl`
* `/data/wikipedia/finetune/val.jsonl`