# Model weights

Here you should find the pre-trained model weights for BERT-base trained on LXMERT and Wikipedia data as well as CLIP-BERT weights in their respective folders. Note that none of the weights for the adaptations can be found here, as they are available under [adaptations/data/runs](adaptations/data/runs).

This folder should contain the following files for the evaluations on GLUE and Visual Property Norms to work:
* `bert-lxmert-trained/mp_rank_00_model_states.pt`
* `bert-lxmert-trained-scratch/mp_rank_00_model_states.pt`
* `bert-wikipedia-trained/mp_rank_00_model_states.pt`
* `clip-bert/mp_rank_00_model_states.pt`

You can either obtain these model weights by performing the training as described in [models/README.md](models/README.md) or by downloading them from Huggingface, as described below.

## Download trained model weights from Huggingface

All of the trained model weights for trained BERT-base models and CLIP-BERT can be found under [this](https://huggingface.co/Lo/measure-visual-commonsense-knowledge-model-weights) Huggingface repo.