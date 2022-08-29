# Models

This folder contains code necessary the evaluated models and their trained model weights. The code encompasses both model code, and code for creating the BERT-base baselines (`trained-LXMERT`, `trained-LXMERT-scratch` and `trained-Wikipedia`).

## Trained model weights
Model weights for the BERT-base variations `trained-LXMERT`, `trained-LXMERT-scratch` and `trained-Wikipedia` should be found in the `data/model-weights` folder. You can download these necessary model weights by following the instructions provided in [models/data/model-weights/README.md](models/data/model-weights/README.md). A description of how to train the models can be found below, should you wish to train your own model weights.

### BERT-base trained-LXMERT

This section describes the experiment of training a BERT base uncased model on the language part of the LXMERT training data from pre-trained BERT weights.

The model can be trained using the following command:

```
models/data/runs/bert-lxmert-train/run.sh
```

We trained the model on four NVIDIA Tesla T4 GPU with 16GB RAM and 8 cores for 9h. The model appeared to have converged at this time.

### BERT-base trained-LXMERT-scratch

This section describes the experiment of training a BERT base uncased model on the language part of the LXMERT training data from scratch.

The model can be trained using the following command:

```
models/data/runs/bert-lxmert-train-scratch/run.sh
```

We trained the model on four NVIDIA Tesla T4 GPU with 16GB RAM and 8 cores for 9h. The model appeared to have converged at this time.

### BERT-base trained-Wikipedia

This section describes the experiment of training a BERT base uncased model on Wikipedia training data.

The model can be trained using the following command:

```
models/data/runs/bert-wikipedia-train/run.sh
```

## CLIP-BERT

This section describes the experiment of training a CLIP-BERT model from pre-trained BERT weights.

First, download the images and corpora for Conceptual Captions, SBU Captions, COCO and Visual Genome QA. Then, create the training data files `models/data/clip-bert/VLP/train.jsonl`, `models/data/clip-bert/VLP/val.jsonl` and `models/data/clip-bert/VLP/clip_features.hdf5` using the makefile `models/data/clip-bert/VLP.mk`. 

The model can then be trained using the following command:

```
models/data/runs/clip-bert/run.sh
```

We trained the model on four NVIDIA Tesla T4 GPU with 16GB RAM and 8 cores for 16h. The model appeared to have converged at this time.

### Download existing datasets for CLIP-BERT

We have made the CLIP image features and corpora available in [this](TBD) Huggingface repo.

## Evaluating the trained models

The model weights used for evaluation can then be found under the respective model folders under `models/data/runs`. Additionally, released model weights may be provided in `models/data/model-weights`. You need to change the model paths in all evaluation notebooks if you do not wish to (or cannot) use the already provided model weights.


## Additional model code

The code for modelling CLIP-BERT can be found under `src/clip_bert` and code for altering LXMERT to work with unimodal language input can be found under `src/lxmert`.
