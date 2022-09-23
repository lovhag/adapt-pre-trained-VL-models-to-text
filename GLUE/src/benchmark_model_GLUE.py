import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    LxmertConfig,
    CLIPModel,
    CLIPTokenizerFast
)

from datetime import datetime
import copy
import json

from models.src.lxmert.alterations import LxmertLanguageOnlyXLayer
from GLUE.src.modeling_lxmert import (
    LxmertForSequenceClassification, 
    get_lxmert_batch, 
    LxmertConfigForSequenceClassification,
    DataCollatorWithPaddingSkippingFeatures
)
from GLUE.src.modeling_visualbert import (
    VisualBertForSequenceClassification, 
    get_visualbert_batch, 
    VisualBertConfigForSequenceClassification
)
from GLUE.src.modeling_clipbert import (
    ClipBertForSequenceClassification, 
    get_clipbert_batch,
)
from GLUE.src.modeling_flava import (
    FlavaForSequenceClassification,
    FlavaConfigForSequenceClassification
)


GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TASKS_WITH_BINARY_LABELS = ["cola", "mrpc", "qqp", "sst2", "wnli"]

GLUE_EVAL_METRICS = {
                        "cola": "matthews_correlation",
                        "mnli": "accuracy",
                        "mrpc": "combined_score",
                        "qnli": "accuracy",
                        "qqp": "combined_score",
                        "rte": "accuracy",
                        "sst2": "accuracy",
                        "stsb": "combined_score",
                        "wnli": "accuracy",
}



logger = logging.getLogger(__name__)

def benchmark_on_GLUE_task(model_name: str, 
                           model_path: str, 
                           tokenizer_name: str, 
                           task_name: str, 
                           train_batch_size: int, 
                           eval_batch_size: int, 
                           epochs: int, lr: float, 
                           weight_decay: float, 
                           cache_dir: str, 
                           max_train_samples: int, 
                           logdir: str, 
                           tb_logdir: str, 
                           dataloader_num_workers: int, 
                           do_train: bool, 
                           do_eval: bool, 
                           do_predict: bool, 
                           visual_features_path: str,
                           visual_boxes_path: str,
                           model_weights_path: str,
                           use_imagined_visual_feats: bool):
    
    def standard_preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=config.max_position_embeddings, truncation=True)

        return result

    def preprocess_function_with_clip(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=config.max_position_embeddings, truncation=True)
        # Add CLIP features
        with torch.no_grad():
            inputs = clip_tokenizer(*args, return_tensors="pt", padding=True, truncation=True).to(device)
            img_feats = clip_model.get_text_features(**inputs).unsqueeze(1).tolist()
        result.update(
            {
                "img_feats": img_feats
            }
        )

        return result

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    
    now = datetime.now()
    model_name_string = model_name.replace("/","-") if model_name else model_path.replace("/","-")
    output_dirname = "GLUE-benchmark-"+task_name+"-"+model_name_string+now.strftime("-%Y-%m-%dT%H-%M")
    output_dirname = os.path.join(logdir, output_dirname)
    os.mkdir(output_dirname)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger.info(f"Benchmarking on GLUE started.")

    # Downloading and loading a dataset from the hub.
    logger.info(f"Loading GLUE task {task_name} datasets")
    datasets = load_dataset("glue", task_name, cache_dir=cache_dir)
    if task_name == "mnli":
        logger.info(f"Also loading GLUE task ax datasets (compatible with mnli)")
        datasets["ax"] = load_dataset("glue", "ax", cache_dir=cache_dir)["test"]

    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    batch_processor = None
    model_name_prefix = model_name.split("-")[0]
    if model_name_prefix == "lxmert":
        config = LxmertConfigForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path or model_name,
            num_labels=num_labels,
            finetuning_task=task_name,
            cache_dir=cache_dir
        )
        model = LxmertForSequenceClassification.from_pretrained(
            model_path or model_name,
            config=config,
            cache_dir=cache_dir,
        )
        if visual_features_path is None:
            prev_encoder = copy.deepcopy(model.lxmert.encoder)
            model.lxmert.encoder.x_layers = torch.nn.ModuleList([LxmertLanguageOnlyXLayer(model.lxmert.encoder.config) for _ in range(model.lxmert.encoder.config.x_layers)])
            model.lxmert.encoder.load_state_dict(prev_encoder.state_dict())

            logger.info(f"Using no visual features for multimodal model.")
            visual_feats = None
            visual_boxes = None
        else:
            assert visual_boxes_path is not None, "For LXMERT both visual features and visual boxes need to be given"
            logger.info(f"Using given visual features for multimodal model.")
            visual_feats = torch.load(visual_features_path)
            visual_boxes = torch.load(visual_boxes_path)

        batch_processor = lambda batch: get_lxmert_batch(batch, visual_feats=visual_feats, visual_boxes=visual_boxes)
        multimodal_features_to_skip = ("visual_feats", "visual_pos")
    elif model_name_prefix == "visualbert":
        config = VisualBertConfigForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path or model_name,
            num_labels=num_labels,
            finetuning_task=task_name,
            cache_dir=cache_dir
        )
        model = VisualBertForSequenceClassification.from_pretrained(
            model_path or model_name,
            config=config,
            cache_dir=cache_dir,
        )

        if visual_features_path is not None:
            logger.info(f"Using given visual features for multimodal model.")
            visual_feats = torch.load(visual_features_path)
        else:
            logger.info(f"Using no visual features for multimodal model.")
            visual_feats = None

        batch_processor = lambda batch: get_visualbert_batch(batch, visual_feats=visual_feats)
        multimodal_features_to_skip = ("visual_embeds", "visual_token_type_ids", "visual_attention_mask", "labels")
    elif model_name_prefix == "clipbert":
        config = BertConfig.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased", 
            num_labels=num_labels,
            finetuning_task=task_name,
            cache_dir=cache_dir
        )
        model = ClipBertForSequenceClassification(config)
        assert model_weights_path is not None, "CLIP-BERT needs to be initialized from pre-trained weights"

        if use_imagined_visual_feats:
            logger.info(f"Using CLIP generated visual features for multimodal model.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
            #clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            batch_processor = lambda batch: get_clipbert_batch(batch, visual_feats=None, use_imagined_visual_feats=True)
        elif visual_features_path is not None:
            logger.info(f"Using given visual features for multimodal model.")
            visual_feats = torch.load(visual_features_path)
            batch_processor = lambda batch: get_clipbert_batch(batch, visual_feats=visual_feats)
        else:
            logger.info(f"Using no visual features for multimodal model.")
            batch_processor = lambda batch: get_clipbert_batch(batch, visual_feats=None)

        multimodal_features_to_skip = ("img_feats")
    elif model_name_prefix == "flava":
        config = FlavaConfigForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path or model_name,
            num_labels=num_labels,
            finetuning_task=task_name,
            cache_dir=cache_dir
        )
        model = FlavaForSequenceClassification.from_pretrained(
            model_path or model_name,
            config=config,
            cache_dir=cache_dir,
        )
    else:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_path or model_name,
            num_labels=num_labels,
            finetuning_task=task_name,
            cache_dir=cache_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path or model_name,
            config=config,
            cache_dir=cache_dir,
        )

    if model_weights_path is not None:
        logger.info(f"Loading provided model weights from '{model_weights_path}'...")
        weights = torch.load(model_weights_path, map_location="cpu")
        if "module" in weights:
            # TODO: write warning message about weights not loaded?
            model.load_state_dict(weights["module"], strict=False)
        else:
            model.load_state_dict(weights)

    # all models use the same tokenizer, except for FLAVA
    if model_name_prefix == "flava":
        tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name or model_path,
            cache_dir=cache_dir,
        )
    if use_imagined_visual_feats:
        preprocess_function = preprocess_function_with_clip
    else: 
        preprocess_function = standard_preprocess_function

    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)

    train_dataset = datasets["train"]
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = datasets["validation_matched" if task_name == "mnli" else "validation"]
    test_dataset = datasets["test_matched" if task_name == "mnli" else "test"]

    eval_tasks = [task_name]
    eval_datasets = [eval_dataset]
    if task_name == "mnli":
        eval_tasks.append("mnli-mm")
        eval_datasets.append(datasets["validation_mismatched"])

    predict_tasks = [task_name]
    predict_datasets = [test_dataset]
    if task_name == "mnli":
        predict_tasks.append("mnli-mm")
        predict_datasets.append(datasets["test_mismatched"])

        # also add the GLUE diagnostics test which suits the MNLI head
        predict_tasks.append("ax")
        predict_datasets.append(datasets["ax"])

    data_collator = None

    # adapt data to multimodal models
    if batch_processor is not None:
        train_dataset.set_transform(batch_processor)
        eval_dataset.set_transform(batch_processor)
        test_dataset.set_transform(batch_processor)

        # also need to transform datasets later used for evaluation and prediction
        for dataset in eval_datasets:
            dataset.set_transform(batch_processor)
        for dataset in predict_datasets:
            dataset.set_transform(batch_processor)

        data_collator = DataCollatorWithPaddingSkippingFeatures(tokenizer, features_to_skip = multimodal_features_to_skip)

    logger.info(f"Datasets loaded.")
    logger.info(f"train dataset contains {len(train_dataset)} examples")
    logger.info(f"eval dataset contains {len(eval_dataset)} examples")
    logger.info(f"test dataset contains {len(test_dataset)} examples")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function (is implicitly used in `compute_metrics`)
    metric = load_metric("glue", task_name)

    num_devices = torch.cuda.device_count()
    per_device_train_batch_size = int(train_batch_size/num_devices)
    per_device_eval_batch_size = int(eval_batch_size/num_devices)

    total_steps_per_epoch = len(train_dataset)/train_batch_size
    logging_steps = int(total_steps_per_epoch/50)+1
    eval_steps = int(total_steps_per_epoch/10)+1
    logger.info(f"number of train steps per train batch: {total_steps_per_epoch}")
    logger.info(f"will log at every {logging_steps} train steps")
    logger.info(f"will evaluate and checkpoint at every {eval_steps} train steps")

    training_args = TrainingArguments(output_dir=output_dirname,
                                        do_eval=True,
                                        evaluation_strategy="steps",
                                        eval_steps=eval_steps,
                                        save_steps=eval_steps,
                                        logging_strategy="steps",
                                        logging_steps=logging_steps,
                                        per_device_train_batch_size=per_device_train_batch_size,
                                        per_device_eval_batch_size=per_device_eval_batch_size,
                                        learning_rate=lr,
                                        weight_decay=weight_decay,
                                        num_train_epochs=epochs,
                                        logging_dir=output_dirname,
                                        save_total_limit=1,
                                        load_best_model_at_end=True,
                                        metric_for_best_model=GLUE_EVAL_METRICS[task_name],
                                        greater_is_better=True,
                                        dataloader_num_workers=dataloader_num_workers,
                                        label_names=["labels"])

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train
    if do_train:
        logger.info(f"*** Training on {task_name} task *** ")
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # Evaluate
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    if do_eval:
        logger.info(f"*** Evaluating on evaluation set of {task_name} task *** ")

        for eval_dataset, task in zip(eval_datasets, eval_tasks):
            logger.info(f"\t*** Evaluating on {task} *** ")
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            metrics["eval_samples"] = len(eval_dataset)

            logname = ("_").join([task.replace("-","_"), "eval"])
            trainer.log_metrics(logname, metrics)
            trainer.save_metrics(logname, metrics)
            #path = os.path.join(output_dirname, task.replace("-", "_"), "_eval_results.json")
            #with open(path, "w") as f:
            #    json.dump(metrics, f, indent=4, sort_keys=True)

    # Predict for test set
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    if do_predict:
        for predict_dataset, task in zip(predict_datasets, predict_tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset.remove_columns_("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(output_dirname, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"*** Creating predictions for test set of {task} task *** ")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            item = min(item, 5) # values should be between 0 and 5
                            item = max(item, 0)
                            writer.write(f"{index}\t{item:3.3f}\n")
                        elif task_name in TASKS_WITH_BINARY_LABELS:
                            writer.write(f"{index}\t{item}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    logger.info("*** DONE! ***")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tokenizer-name", default="bert-base-uncased")
    parser.add_argument("--task-name", default="cola")
    parser.add_argument("--train-batch-size", default=32, type=int) # 16, 32
    parser.add_argument("--eval-batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=3, type=int) # 2, 3, 4
    parser.add_argument("--lr", default=5e-5, type=float) # TODO: Potentially fine-tune between 5e-5, 3e-5, 2e-5
    parser.add_argument("--weight-decay", default=0.01, type=float) # used by BERT authors
    parser.add_argument("--cache-dir", default="GLUE_cache")
    parser.add_argument("--max-train-samples", default=None, type=int)
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--tb-logdir", default="tb-logs")
    parser.add_argument("--dataloader-num-workers", default=4, type=int)
    parser.add_argument('--do-train', default=False, action='store_true')
    parser.add_argument('--do-eval', default=False, action='store_true')
    parser.add_argument('--do-predict', default=False, action='store_true')
    parser.add_argument("--visual-features-path", default=None)
    parser.add_argument("--visual-boxes-path", default=None)
    parser.add_argument("--model-weights-path", default=None)
    parser.add_argument('--use-imagined-visual-feats', default=False, action='store_true')

    args = parser.parse_args()

    benchmark_on_GLUE_task(
        model_name=args.model_name, 
        model_path=args.model_path,
        tokenizer_name=args.tokenizer_name,
        task_name=args.task_name, 
        train_batch_size=args.train_batch_size, 
        eval_batch_size=args.eval_batch_size, 
        epochs=args.epochs, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        cache_dir=args.cache_dir,
        max_train_samples=args.max_train_samples,
        logdir=args.logdir,
        tb_logdir=args.tb_logdir,
        dataloader_num_workers=args.dataloader_num_workers,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        visual_features_path=args.visual_features_path,
        visual_boxes_path=args.visual_boxes_path,
        model_weights_path=args.model_weights_path,
        use_imagined_visual_feats=args.use_imagined_visual_feats
    )