# Finetune the VL models to a text-only dataset

Fine-tuning is performed using the run scripts `run.sh` for each model (CLIP-BERT, LXMERT or VisualBERT) and finetuning text data (Wikipedia or LXMERT train data). 

The necessary finetuning data should have been generated as described in [data/README.md](data/README.md) before running the scripts in this folder.

## Download finetuned model weights 

The finetuned model weights can be downloaded from [this](https://huggingface.co/Lo/adapt-pre-trained-VL-models-to-text-finetuned-weights) Huggingface repo. Make sure that the model weights are put under the respective model run folders, or filepaths will need to be changed in the evaluation code for GLUE and Visual Property Norms.