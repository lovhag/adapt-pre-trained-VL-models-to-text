# Evaluate on GLUE

A model is evaluated on GLUE with the [src/benchmark_model_GLUE.py](src/benchmark_model_GLUE.py) script. The arguments used to generate the GLUE results presented in the article can be found in the run scripts [data/runs](data/runs) for each model and adaptation. Each run script assumes that you are standing in the root folder of this repo.

In the run files, an array ID is used to indicate what GLUE task to evaluate on. The IDs are as follows:

| Task | Array id | Comment |
| ---- | -------- | ------- |
| AX | 1 | Has no train set and is evaluated with the same head as for MNLI. |
| CoLA | 0 | |
| MNLI-m | 1 | |
| MNLI-mm | 1 | |
| MRPC | 2 | |
| QNLI | 3 | |
| QQP | 4 | |
| RTE | 5 | |
| SST-2 | 6 | |
| STSB | 7 | |
| WNLI | 8 | |

Fine-tuning and evaluating for one GLUE task with one model and its adaptation takes between 6 minutes and 2.5 hours on two NVIDIA Tesla A100 GPU with 40GB RAM, depending on the task. QQP and MNLI generally take the longest time to fine-tune and evaluate on. 

Note that the GLUE folder can consume as much as 830 GB of memory when evaluating all of the 29 model and adaptation configurations on the 9 different GLUE tasks. This comes from Huggingface storing 36 GB in cache for the GLUE evaluations and 792 GB being stored for model checkpoints. If you only want the final results you can reduce this memory consumption by gradually deleting model checkpoints as you go along with the model evaluations.

## Plotting GLUE evaluation results

Once each desired model and adaptation has been trained and evaluated with the script mentioned above, the results can be collected in [GLUE/src/generate-results.ipynb](GLUE/src/generate-results.ipynb) and then plotted in [GLUE/src/generate-plots.ipynb](GLUE/src/generate-plots.ipynb).

## Reproducing GLUE results

The final GLUE evaluation results will differ slightly for each run, due to the randomness in the fine-tuning on the tasks. However, these differences in results do not change our previous main conclusions.

## Submitting test predictions to GLUE

The prediction files for each GLUE task can also be collected in a format corresponding to the one expected by the GLUE submission board by running 
```
src/collect_model_predictions.sh <folder-with-all-model-logs> <sumbission-folder-name>
```

Then create the submission for GLUE using the following command in the folder with the model predictions.
```
zip -r submission.zip *.tsv
```
