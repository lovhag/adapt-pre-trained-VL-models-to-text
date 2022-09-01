# Visual Property Norms

This is derivative work of the CSLB Norms and the CSLB, those who carried out the original collection of data in the CSLB Norms and the funder/s or sponsor/s of the research bear no responsibility for the further analysis or interpretation of this.

This folder contains all the instructions and code necessary for reproducing the Visual Property Norms results. The main code is splitted over the following files:
* [create-visual-property-norms-evaluation.ipynb](create-visual-property-norms-evaluation.ipynb) creates the [queries](data/queries) used for the Visual Property Norms evaluation. It also creates the concepts and features listed in [pf-partitions](data/pf-partitions). To be able to run this notebook, you need to have downloaded the CSLB property norms datafile `norms.dat` to the [data folder](data). Access to the data can be requested for via the [CSLB website](https://cslb.psychol.cam.ac.uk/propnorms#:~:text=The%20Centre%20for%20Speech%2C%20Language,feature%20representations%20of%20conceptual%20knowledge), since we are not allowed to distribute it.
* [generate-results.ipynb](generate-results.ipynb) is used to generate the model results on Visual Property Norms found under `data/results/results.csv`. To be able to generate all necessary results you need to have the model weights and adaptations data obtained via [models](models) and [adaptations](adaptations) respectively.
* [generate-plots.ipynb](generate-plots.ipynb) is used to generate the plots corresponding to the results generated above.

## Reproducing our results

You should be able to reproduce our results completely by running [generate-results.ipynb](generate-results.ipynb). Some slight differences in model scores can occur, while they are in the range of 0.0001 units, so the final plot should be exactly the same as ours. 

Evaluating one model with one adaptation takes approximately 4 minutes (9 mins for LXMERT) on a NVIDIA Tesla A100 GPU with 40GB RAM, so running all of the 29 model and adaptation configurations should take around 2 hours.

## How to evaluate your own model

A model is evaluated through the following code block in the [generate-results.ipynb](generate-results.ipynb) notebook:

```python
model_name = [insert model name here]
tokenizer = [insert your model tokenizer here]
model = [insert the model you wish to evaluate here]
model.eval()
   
get_preds = [define a function for how to get predicitons from your model with only a list of `questions` as input]

query_files = os.listdir(QUERIES_FOLDER) # this is predefined
results = update_results_with_model_preds(results, get_preds, query_files, tokenizer)
```

To evaluate your own model, you simply need to replace the parts in [brackets] with your model information. Most important is to define the `get_preds` function that takes a list of questions as input and returns a list of model predictions based on these questions. The necessary queries and functions to measure the metrics are already provided.

