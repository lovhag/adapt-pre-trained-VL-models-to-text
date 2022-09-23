import json
import numpy as np
import os
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch.utils.data import DataLoader
import copy
import istarmap  # import to apply patch for tqdm to work
from multiprocessing import Pool
from tqdm import tqdm

import transformers
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

from transformers import (
    BertTokenizer, 
    BertForMaskedLM, 
    BertConfig, 
    CLIPModel, 
    CLIPProcessor, 
    FlavaForPreTraining,
    VisualBertForPreTraining, 
    VisualBertConfig, 
    LxmertForPreTraining, 
    LxmertConfig
)

from models.src.clip_bert.modeling_bert import BertImageForMaskedLM
from models.src.lxmert.alterations import LxmertLanguageOnlyXLayer

with open("visual_property_norms/data/labels.txt", "r") as f:
    MASK_LABELS = [label.strip() for label in f.readlines()]

def get_model_preds_for_questions(questions, model, tokenizer, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = DataLoader(questions, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        preds = []
        for questions_batch in iter(dataloader):
            inputs = tokenizer(questions_batch, return_tensors="pt", padding=True).to(device)
            mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)[:, 1][:, None]
            assert len(mask_idx) == inputs.input_ids.shape[0], "ERROR: Found multiple [MASK] tokens per example"

            outputs = model(**inputs)["logits"] if "logits" in model(**inputs) else model(**inputs)["prediction_logits"]
            pred = outputs.gather(1, mask_idx.repeat(1, outputs.shape[-1]).unsqueeze(1)).squeeze(1)
            preds.append(pred)

    preds = torch.cat(preds)
    return preds

def get_map_score_for_preds(labels, pred, tokenizer):
    scores = []
    assert pred[0].shape[0] == tokenizer.vocab_size
    vocab = tokenizer.get_vocab()
    for query_ix in range(len(labels)):
        y_true = [0]*tokenizer.vocab_size
        for label in labels[query_ix]:
            y_true[vocab[label]] = 1 
        scores.append(average_precision_score(y_true, pred[query_ix]))
    
    return scores

def get_map_score_for_masked_preds(labels, pred, tokenizer, mask_labels):
    scores = []
    assert pred[0].shape[0] == tokenizer.vocab_size
    vocab = tokenizer.get_vocab()
    mask_ix = [vocab[mask_label] for mask_label in mask_labels]
    
    for query_ix in range(len(labels)):
        y_true = [0]*len(mask_ix)
        for label in labels[query_ix]:
            y_true[mask_ix.index(vocab[label])] = 1 
        scores.append(average_precision_score(y_true, pred[query_ix][mask_ix]))
    
    return scores

def visualize_predictions(pred, questions, labels, tokenizer, num):
    random_ix = np.random.choice(len(pred), num, replace=False)
    for i in random_ix:
        print("-------------------------------")
        print(f"Question: {questions[i]}")
        print(f"Golden labels: {labels[i]}")
        print(f"Predicted labels: {tokenizer.decode(pred[i].topk(k=20).indices)}")
        print("-------------------------------")

def get_model_results_per_query_file(query_file, run_config):
    # load the model
    model_name = run_config["model_name"].lower()
    if model_name=="bert-base":
        model = BertForMaskedLM.from_pretrained(run_config["model_path"])
    elif model_name=="clip-bert":
        model, clip_model, clip_processor = get_clip_bert_model(run_config["model_path"], no_visual_prediction=run_config["no_visual_prediction"])
    elif model_name=="lxmert":
        model = LxmertForPreTraining.from_pretrained(run_config["model_path"])
        # if we are using LXMERT without visual features, we need to change the model slightly
        if run_config["visual_features_path"] is None:
            prev_encoder = copy.deepcopy(model.lxmert.encoder)
            model.lxmert.encoder.x_layers = torch.nn.ModuleList([LxmertLanguageOnlyXLayer(model.lxmert.encoder.config) for _ in range(model.lxmert.encoder.config.x_layers)])
            model.lxmert.encoder.load_state_dict(prev_encoder.state_dict())
    elif model_name=="visualbert":
        model = VisualBertForPreTraining.from_pretrained(run_config["model_path"])
    elif model_name=="flava":
        model = FlavaForPreTraining.from_pretrained(run_config["model_path"])
    else:
        raise ValueError(f"model_name {model_name} not recognized")

    # potentially load model weights on top of given model configuration
    if run_config["model_weights_path"] is not None:
        model.load_state_dict(torch.load(run_config["model_weights_path"], map_location="cpu")["module"], strict=False)

    model.eval()
    if model_name=="flava":
        tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # load potential given features
    if run_config["visual_features_path"] is not None:
        visual_features = torch.load(run_config["visual_features_path"]).cpu()
    else:
        visual_features = None

    if run_config["visual_boxes_path"] is not None:
        visual_boxes = torch.load(run_config["visual_boxes_path"]).cpu()
    else:
        visual_boxes = None

    # extract data from query file 
    with open(query_file) as f:
        examples = [json.loads(line) for line in f.readlines()]

    questions = [ex["query"] for ex in examples]
    labels = [ex["labels"] for ex in examples]
    concepts = [ex["concept"] for ex in examples]
    feature_starters = [ex["feature_starter"] for ex in examples]

    # get predictions from model
    if model_name=="bert-base":
        pred = get_model_preds_for_questions(questions, model, tokenizer, run_config["batch_size"])
    elif model_name=="clip-bert":
        pred = get_clip_bert_preds_for_questions(questions, model, clip_model, clip_processor, tokenizer, run_config["no_visual_prediction"], run_config["batch_size"], visual_features)
    elif model_name=="lxmert":
        pred = get_lxmert_preds_for_questions(questions, model, tokenizer, run_config["batch_size"], visual_features, visual_boxes)
    elif model_name=="visualbert":
        pred = get_visualbert_preds_for_questions(questions, model, tokenizer, run_config["batch_size"], visual_features)
    elif model_name=="flava":
        if "input_feats" in run_config:
            assert "output_feats" in run_config, "If 'input_feats' are defined for FLAVA, 'output_feats' should be as well."
            pred = get_flava_preds_for_questions(questions, model, tokenizer, run_config["batch_size"], run_config["input_feats"], run_config["output_feats"])
        else:
            pred = get_flava_preds_for_questions(questions, model, tokenizer, run_config["batch_size"])

    # evaluate model predictions
    scores = get_map_score_for_preds(labels, pred.cpu().detach().numpy(), tokenizer)
    masked_scores = get_map_score_for_masked_preds(labels, pred.cpu().detach().numpy(), tokenizer, MASK_LABELS)    
    mean_nbr_alternatives = np.mean([len(alternatives) for alternatives in labels])

    query_template = examples[0]["query_template"] #same for the same file
    pf = examples[0]["pf"]
    
    # format results nicely
    results = pd.DataFrame()
    for query_ix in range(len(labels)):
        results_entry = {"concept": concepts[query_ix],
                            "query_template": query_template, 
                            "feature_starter": feature_starters[query_ix],
                            "pf": pf,
                            "score": scores[query_ix], 
                            "masked_score": masked_scores[query_ix], 
                            "nbr_alternatives": len(labels[query_ix]),
                            "top10_preds": tokenizer.convert_ids_to_tokens(pred[query_ix].topk(k=10).indices),
                            "gold_labels": labels[query_ix]}
        results = results.append(results_entry, ignore_index=True).reset_index(drop=True)
    
    return results

# Structure for input argument:
# run_config = {
#                 "model_name": ,
#                 "model_path": ,
#                 "model_weights_path": ,
#                 "batch_size": ,
#                 "visual_features_path": ,
#                 "visual_boxes_path": ,
#                 "no_visual_prediction": 
# }

def get_model_results(queries_folder, run_config, max_pool=4):
    # create arguments for each process to parallelize
    query_files = [os.path.join(queries_folder, q_file) for q_file in os.listdir(queries_folder) if q_file.endswith(".jsonl")]
    iterable_arguments = [(query_file, run_config) for query_file in query_files]

    # run processes
    with Pool(max_pool) as p:
        pool_outputs = list(
                tqdm(p.istarmap(get_model_results_per_query_file, iterable_arguments),
                total=len(iterable_arguments))
        )    

    return pd.concat(pool_outputs)


def get_save_filename(model_name, adaptation):
    return ("-").join([model_name, adaptation])+".csv"

## CLIP-BERT
def get_clip_bert_model(bert_image_model_path: str, no_visual_prediction: bool=False):
    # Load BertImageForMaskedLM model
    config = BertConfig.from_pretrained("bert-base-uncased")
    bert_image_model = BertImageForMaskedLM(config).eval()
    bert_image_model.load_state_dict(torch.load(bert_image_model_path, map_location="cpu")["module"], strict=False)
    bert_image_model.eval()

    # Load CLIP
    if not no_visual_prediction:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        return bert_image_model, clip_model, clip_processor
    else:
        return bert_image_model, None, None

def get_clip_bert_preds_for_questions(questions,
                                      model, 
                                      clip_model, 
                                      clip_processor,
                                      tokenizer,
                                      no_visual_prediction: bool=False,
                                      batch_size=64,
                                      visual_features=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    clip_model = clip_model.to(device).eval() if clip_model is not None else clip_model
    
    dataloader = DataLoader(questions, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        preds = []
        for questions_batch in iter(dataloader):
            inputs = tokenizer(questions_batch, return_tensors="pt", padding=True).to(device)
            
            # Given visual features takes precedence
            if visual_features is not None:
                img_feats = torch.as_tensor(np.tile(visual_features, (len(questions_batch), 1))).to(device).unsqueeze(1)
                inputs["img_feats"] = img_feats
            # Predict visual features using CLIP
            elif not no_visual_prediction:
                img_feats = clip_model.get_text_features(**clip_processor(text=questions_batch, return_tensors="pt", padding=True).to(device)).unsqueeze(1)
                inputs["img_feats"] = img_feats
                
            outputs = model(**inputs)["logits"]
            
            mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)[:, 1][:, None]
            assert len(mask_idx) == inputs.input_ids.shape[0], "ERROR: Found multiple [MASK] tokens per example"
            pred = outputs.gather(1, mask_idx.repeat(1, outputs.shape[-1]).unsqueeze(1)).squeeze(1)
            preds.append(pred)
        preds = torch.cat(preds)

    return preds

## LXMERT
LXMERT_FEATURES_SHAPE = (1, 2048)
LXMERT_NORMALIZED_BOXES_SHAPE = (1, 4)

def get_lxmert_preds_for_questions(questions, model, tokenizer, batch_size=64, visual_features=None, visual_boxes=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = DataLoader(questions, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        preds = []
        for questions_batch in iter(dataloader):
            inputs = tokenizer(questions_batch, return_tensors="pt", padding=True).to(device)
            mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)[:, 1][:, None]
            assert len(mask_idx) == inputs.input_ids.shape[0], "ERROR: Found multiple [MASK] tokens per example"
            
            # align the visual inputs
            batch_size = len(questions_batch)
            if visual_features is None:
                visual_feats = torch.empty((batch_size,)+LXMERT_FEATURES_SHAPE).uniform_(0, 10).to(device)
            else:
                visual_feats = torch.as_tensor(np.tile(visual_features, (batch_size, 1, 1))).to(device)
            if visual_boxes is None:
                visual_pos = torch.empty((batch_size,)+LXMERT_NORMALIZED_BOXES_SHAPE).uniform_(0, 1).to(device)
            else:
                visual_pos = torch.as_tensor(np.tile(visual_boxes, (batch_size, 1, 1))).to(device)

            inputs.update({
                "visual_feats": visual_feats,
                "visual_pos": visual_pos
            })
            outputs = model(**inputs)["logits"] if "logits" in model(**inputs) else model(**inputs)["prediction_logits"]
            pred = outputs.gather(1, mask_idx.repeat(1, outputs.shape[-1]).unsqueeze(1)).squeeze(1)
            preds.append(pred)

    preds = torch.cat(preds)
    return preds


def get_visualbert_preds_for_questions(questions, model, tokenizer, batch_size=64, visual_features=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = DataLoader(questions, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        preds = []
        for questions_batch in iter(dataloader):
            inputs = tokenizer(questions_batch, return_tensors="pt", padding=True).to(device)
            mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)[:, 1][:, None]
            assert len(mask_idx) == inputs.input_ids.shape[0], "ERROR: Found multiple [MASK] tokens per example"
            
            # align the visual inputs
            batch_size = len(questions_batch)
            if visual_features is not None:
                visual_embeds = torch.as_tensor(np.tile(visual_features, (batch_size, 1, 1))).to(device)
                visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(device)
                inputs.update(
                    {
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                    }
                )

            outputs = model(**inputs)["logits"] if "logits" in model(**inputs) else model(**inputs)["prediction_logits"]
            pred = outputs.gather(1, mask_idx.repeat(1, outputs.shape[-1]).unsqueeze(1)).squeeze(1)
            preds.append(pred)

    preds = torch.cat(preds)
    return preds

def get_flava_preds_for_questions(questions, model, tokenizer, batch_size, input_feats="input_ids", output_feats="mlm_logits"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = DataLoader(questions, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        preds = []
        for questions_batch in iter(dataloader):
            inputs = tokenizer(questions_batch, return_tensors="pt", padding="max_length", max_length=77).to(device)
            mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)[:, 1][:, None]
            assert len(mask_idx) == inputs.input_ids.shape[0], "ERROR: Found multiple [MASK] tokens per example"

            if input_feats=="input_ids":
                assert "input_ids" in inputs
            elif input_feats=="input_ids_masked":
                inputs["input_ids_masked"] = inputs.pop("input_ids")
            else:
                raise ValueError(f"Did not recognize input_feats configuration '{input_feats}'")

            outputs = model(**inputs)[output_feats]
            pred = outputs.gather(1, mask_idx.repeat(1, outputs.shape[-1]).unsqueeze(1)).squeeze(1)
            preds.append(pred)

    preds = torch.cat(preds)
    return preds