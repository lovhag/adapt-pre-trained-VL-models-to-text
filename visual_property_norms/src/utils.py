import json
import numpy as np
import os
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig
from models.src.clip_bert.modeling_bert import BertImageForMaskedLM


QUERIES_FOLDER = "visual_property_norms/data/queries"

with open("visual_property_norms/data/labels.txt", "r") as f:
    MASK_LABELS = [label.strip() for label in f.readlines()]

def get_model_preds_for_questions(model, tokenizer, questions, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dataloader = DataLoader(questions, batch_size=batch_size, shuffle=False, num_workers=4)
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

def get_model_results(get_preds, tokenizer):
    query_files = os.listdir(QUERIES_FOLDER)

    for query_file in tqdm(query_files):
        with open(os.path.join(QUERIES_FOLDER, query_file)) as f:
            examples = [json.loads(line) for line in f.readlines()]

        questions = [ex["query"] for ex in examples]
        labels = [ex["labels"] for ex in examples]
        concepts = [ex["concept"] for ex in examples]
        feature_starters = [ex["feature_starter"] for ex in examples]
        pred = get_preds(questions)
        scores = get_map_score_for_preds(labels, pred.cpu().detach().numpy(), tokenizer)
        masked_scores = get_map_score_for_masked_preds(labels, pred.cpu().detach().numpy(), tokenizer, MASK_LABELS)    
        mean_nbr_alternatives = np.mean([len(alternatives) for alternatives in labels])

        query_template = examples[0]["query_template"] #same for the same file
        pf = examples[0]["pf"]
        
        results = pd.DataFrame()
        for query_ix in range(len(labels)):
            assert len(results[(results.model==model_name) & 
                               (results.concept==concepts[query_ix]) & 
                               (results.feature_starter==feature_starters[query_ix]) & 
                               (results.query_template==query_template) & 
                               (results.pf==pf)]) == 0, "Should not append results to already existing key values"
            results_entry = {"concept": concepts[query_ix],
                             "query_template": query_template, 
                             "feature_starter": feature_starters[query_ix],
                             "pf": pf,
                             "score": scores[query_ix], 
                             "masked_score": masked_scores[query_ix], 
                             "nbr_alternatives": len(labels[query_ix]),
                             "top10_preds": tokenizer.convert_ids_to_tokens(pred[query_ix].topk(k=10).indices),
                             #"top10_preds": [print(val) for val in pred[query_ix].topk(k=10).indices],
                             "gold_labels": labels[query_ix]}
            results = results.append(results_entry, ignore_index=True).reset_index(drop=True)
        
    return results

def get_save_filename(model_name, adaptation):
    return ("-").join([model_name, adaptation])

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

def get_clip_bert_preds_for_questions(model, 
                                      clip_model, 
                                      clip_processor,
                                      questions,
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
                img_feats = visual_features.unsqueeze(0).repeat(len(questions_batch), 1).to(device).unsqueeze(1)
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

def get_lxmert_preds_for_questions(model, tokenizer, questions, batch_size=64, visual_features=None, visual_boxes=None):
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
                visual_feats = visual_features.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
            if visual_boxes is None:
                visual_pos = torch.empty((batch_size,)+LXMERT_NORMALIZED_BOXES_SHAPE).uniform_(0, 1).to(device)
            else:
                visual_pos = visual_boxes.unsqueeze(0).repeat(batch_size, 1, 1).to(device)

            inputs.update({
                "visual_feats": visual_feats,
                "visual_pos": visual_pos
            })
            outputs = model(**inputs)["logits"] if "logits" in model(**inputs) else model(**inputs)["prediction_logits"]
            pred = outputs.gather(1, mask_idx.repeat(1, outputs.shape[-1]).unsqueeze(1)).squeeze(1)
            preds.append(pred)

    preds = torch.cat(preds)
    return preds


def get_visualbert_preds_for_questions(model, tokenizer, questions, batch_size=64, visual_features=None):
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
                visual_embeds = visual_features.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
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